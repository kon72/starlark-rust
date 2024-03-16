/*
 * Copyright 2019 The Starlark in Rust Authors.
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! This file provides an implementation of `LspContext` specifically aimed at
//! the use in a Bazel project. You can invoke it by using `starlark --lsp --bazel`.
//! Note that only `--lsp` mode is supported.
//!
//! This module is temporary, for the purpose of rapid iteration while the LSP
//! interface develops. After the API of the `LspContext` trait stabilizes, this
//! module will be removed, and extracted to its own project.

mod label;

use std::borrow::Cow;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::io;
use std::iter;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use either::Either;
use lsp_types::CompletionItemKind;
use lsp_types::Url;
use starlark::analysis::find_call_name::AstModuleFindCallName;
use starlark::analysis::AstModuleLint;
use starlark::docs::get_registered_starlark_docs;
use starlark::docs::render_docs_as_code;
use starlark::docs::Doc;
use starlark::docs::DocItem;
use starlark::docs::DocModule;
use starlark::environment::FrozenModule;
use starlark::environment::Globals;
use starlark::environment::Module;
use starlark::errors::EvalMessage;
use starlark::eval::Evaluator;
use starlark::syntax::AstModule;
use starlark::syntax::Dialect;
use starlark_lsp::completion::StringCompletionResult;
use starlark_lsp::completion::StringCompletionTextEdit;
use starlark_lsp::completion::StringCompletionType;
use starlark_lsp::error::eval_message_to_lsp_diagnostic;
use starlark_lsp::server::LspContext;
use starlark_lsp::server::LspEvalResult;
use starlark_lsp::server::LspUrl;
use starlark_lsp::server::StringLiteralResult;

use self::label::Label;
use crate::eval::ContextMode;
use crate::eval::EvalResult;

#[derive(Debug, thiserror::Error)]
enum ContextError {
    /// The provided Url was not absolute and it needs to be.
    #[error("Path for URL `{}` was not absolute", .0)]
    NotAbsolute(LspUrl),
    /// The scheme provided was not correct or supported.
    #[error("Url `{}` was expected to be of type `{}`", .1, .0)]
    WrongScheme(String, LspUrl),
}

/// Errors when [`LspContext::resolve_load()`] cannot resolve a given path.
#[derive(thiserror::Error, Debug)]
enum ResolveLoadError {
    /// Attempted to resolve a relative path, but no current_file_path was provided,
    /// so it is not known what to resolve the path against.
    #[error("Relative label `{}` provided, but current_file_path could not be determined", .0)]
    MissingCurrentFilePath(Label),
    /// The scheme provided was not correct or supported.
    #[error("Url `{}` was expected to be of type `{}`", .1, .0)]
    WrongScheme(String, LspUrl),
    /// Received a load for an absolute path from the root of the workspace, but the
    /// path to the workspace root was not provided.
    #[error("Label `{}` is absolute from the root of the workspace, but no workspace root was provided", .0)]
    MissingWorkspaceRoot(Label),
    /// The path contained a repository name that is not known to Bazel.
    #[error("Cannot resolve label `{}` because the repository `{}` is unknown", .0, .1)]
    UnknownRepository(Label, String),
    /// The path contained a target name that does not resolve to an existing file.
    #[error("Cannot resolve path `{}` because the file does not exist", .0)]
    TargetNotFound(String),
}

/// Errors when [`LspContext::render_as_load()`] cannot render a given path.
#[derive(thiserror::Error, Debug)]
enum RenderLoadError {
    /// Attempted to get the filename of a path that does not seem to contain a filename.
    #[error("Path `{}` provided, which does not seem to contain a filename", .0.display())]
    MissingTargetFilename(PathBuf),
    /// The scheme provided was not correct or supported.
    #[error("Urls `{}` and `{}` was expected to be of type `{}`", .1, .2, .0)]
    WrongScheme(String, LspUrl, LspUrl),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FilesystemFileCompletionOptions {
    All,
    OnlyLoadable,
    None,
}

/// Options for resolving filesystem completions.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FilesystemCompletionOptions {
    /// Whether to include directories in the results.
    directories: bool,
    /// Whether to include files in the results.
    files: FilesystemFileCompletionOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FilesystemCompletionResultKind {
    Directory,
    File,
}

/// A possible result in auto-complete for a filesystem context.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FilesystemCompletionResult {
    value: String,
    kind: FilesystemCompletionResultKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TargetKind {
    SourceFile,
    GeneratedFile,
    Rule(String),
    Unknown(String),
}

/// A possible result in auto-complete for a target context.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TargetCompletionResult {
    value: String,
    kind: TargetKind,
}

impl fmt::Display for TargetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TargetKind::SourceFile => f.write_str("source file")?,
            TargetKind::GeneratedFile => f.write_str("generated file")?,
            TargetKind::Rule(ref name) => {
                f.write_str(name)?;
                f.write_str(" rule")?;
            }
            TargetKind::Unknown(ref name) => f.write_str(name)?,
        }

        Ok(())
    }
}

impl TargetKind {
    pub fn parse(kind: &str) -> Self {
        if kind == "source file" {
            TargetKind::SourceFile
        } else if kind == "generated file" {
            TargetKind::GeneratedFile
        } else if let Some(rule_name) = kind.strip_suffix(" rule") {
            TargetKind::Rule(rule_name.to_owned())
        } else {
            TargetKind::Unknown(kind.to_owned())
        }
    }
}

pub(crate) fn main(
    lsp: bool,
    print_non_none: bool,
    is_interactive: bool,
    prelude: &[PathBuf],
    dialect: Dialect,
    globals: Globals,
) -> anyhow::Result<()> {
    if !lsp {
        return Err(anyhow::anyhow!("Bazel mode only supports `--lsp`"));
    }

    // NOTE: Copied from `main.rs`
    let mut ctx = BazelContext::new(
        ContextMode::Check,
        print_non_none,
        prelude,
        is_interactive,
        dialect,
        globals,
    )?;

    ctx.mode = ContextMode::Check;
    starlark_lsp::server::stdio_server(ctx)?;

    Ok(())
}

pub(crate) struct BazelContext {
    pub(crate) workspace_name: Option<String>,
    pub(crate) external_output_base: Option<PathBuf>,
    pub(crate) mode: ContextMode,
    pub(crate) print_non_none: bool,
    pub(crate) prelude: Vec<FrozenModule>,
    pub(crate) module: Option<Module>,
    pub(crate) dialect: Dialect,
    pub(crate) globals: Globals,
    pub(crate) builtin_docs: HashMap<LspUrl, String>,
    pub(crate) builtin_symbols: HashMap<String, LspUrl>,
}

impl BazelContext {
    const DEFAULT_WORKSPACE_NAME: &'static str = "__main__";
    const BUILD_FILE_NAMES: [&'static str; 2] = ["BUILD", "BUILD.bazel"];
    const LOADABLE_EXTENSIONS: [&'static str; 1] = ["bzl"];

    pub(crate) fn new(
        mode: ContextMode,
        print_non_none: bool,
        prelude: &[PathBuf],
        module: bool,
        dialect: Dialect,
        globals: Globals,
    ) -> anyhow::Result<Self> {
        let prelude: Vec<_> = prelude
            .iter()
            .map(|x| {
                let env = Module::new();
                {
                    let mut eval = Evaluator::new(&env);
                    let module =
                        AstModule::parse_file(x, &dialect).map_err(starlark::Error::into_anyhow)?;
                    eval.eval_module(module, &globals)
                        .map_err(starlark::Error::into_anyhow)?;
                }
                env.freeze()
            })
            .collect::<anyhow::Result<_>>()?;

        let module = if module {
            Some(Self::new_module(&prelude))
        } else {
            None
        };
        let mut builtins: HashMap<LspUrl, Vec<Doc>> = HashMap::new();
        let mut builtin_symbols: HashMap<String, LspUrl> = HashMap::new();
        for doc in get_registered_starlark_docs() {
            let uri = Self::url_for_doc(&doc);
            builtin_symbols.insert(doc.id.name.clone(), uri.clone());
            builtins.entry(uri).or_default().push(doc);
        }
        let builtin_docs = builtins
            .into_iter()
            .map(|(u, ds)| (u, render_docs_as_code(&ds)))
            .collect();

        let mut raw_command = Command::new("bazel");
        let mut command = raw_command.arg("info");
        command = command.current_dir(std::env::current_dir()?);

        let output = command.output()?;
        if !output.status.success() {
            return Err(anyhow::anyhow!("Command `bazel info` failed"));
        }

        let output = String::from_utf8(output.stdout)?;
        let mut execroot = None;
        let mut output_base = None;
        for line in output.lines() {
            if let Some((key, value)) = line.split_once(": ") {
                match key {
                    "execution_root" => execroot = Some(value),
                    "output_base" => output_base = Some(value),
                    _ => {}
                }
            }
        }

        Ok(Self {
            mode,
            print_non_none,
            prelude,
            module,
            dialect,
            globals,
            builtin_docs,
            builtin_symbols,
            workspace_name: execroot.and_then(|execroot| {
                match PathBuf::from(execroot)
                    .file_name()?
                    .to_string_lossy()
                    .to_string()
                {
                    name if name == Self::DEFAULT_WORKSPACE_NAME => None,
                    name => Some(name),
                }
            }),
            external_output_base: output_base
                .map(|output_base| PathBuf::from(output_base).join("external")),
        })
    }

    // Convert an anyhow over iterator of EvalMessage, into an iterator of EvalMessage
    fn err(
        file: &str,
        result: starlark::Result<EvalResult<impl Iterator<Item = EvalMessage>>>,
    ) -> EvalResult<impl Iterator<Item = EvalMessage>> {
        match result {
            Err(e) => EvalResult {
                messages: Either::Left(iter::once(EvalMessage::from_error(Path::new(file), &e))),
                ast: None,
            },
            Ok(res) => EvalResult {
                messages: Either::Right(res.messages),
                ast: res.ast,
            },
        }
    }

    fn url_for_doc(doc: &Doc) -> LspUrl {
        let url = match &doc.item {
            DocItem::Module(_) => Url::parse("starlark:/native/builtins.bzl").unwrap(),
            DocItem::Object(_) => {
                Url::parse(&format!("starlark:/native/builtins/{}.bzl", doc.id.name)).unwrap()
            }
            DocItem::Function(_) | DocItem::Property(_) => {
                Url::parse("starlark:/native/builtins.bzl").unwrap()
            }
        };
        LspUrl::try_from(url).unwrap()
    }

    fn new_module(prelude: &[FrozenModule]) -> Module {
        let module = Module::new();
        for p in prelude {
            module.import_public_symbols(p);
        }
        module
    }

    fn go(&self, file: &str, ast: AstModule) -> EvalResult<impl Iterator<Item = EvalMessage>> {
        let mut warnings = Either::Left(iter::empty());
        let mut errors = Either::Left(iter::empty());
        let final_ast = match self.mode {
            ContextMode::Check => {
                warnings = Either::Right(self.check(&ast));
                Some(ast)
            }
            ContextMode::Run => {
                errors = Either::Right(self.run(file, ast).messages);
                None
            }
        };
        EvalResult {
            messages: warnings.chain(errors),
            ast: final_ast,
        }
    }

    fn run(&self, file: &str, ast: AstModule) -> EvalResult<impl Iterator<Item = EvalMessage>> {
        let new_module;
        let module = match self.module.as_ref() {
            Some(module) => module,
            None => {
                new_module = Self::new_module(&self.prelude);
                &new_module
            }
        };
        let mut eval = Evaluator::new(module);
        eval.enable_terminal_breakpoint_console();
        Self::err(
            file,
            eval.eval_module(ast, &self.globals)
                .map(|v| {
                    if self.print_non_none && !v.is_none() {
                        println!("{}", v);
                    }
                    EvalResult {
                        messages: iter::empty(),
                        ast: None,
                    }
                })
                .map_err(Into::into),
        )
    }

    fn check(&self, module: &AstModule) -> impl Iterator<Item = EvalMessage> {
        let globals = if self.prelude.is_empty() {
            None
        } else {
            let mut globals = HashSet::new();
            for modu in &self.prelude {
                for name in modu.names() {
                    globals.insert(name.as_str().to_owned());
                }
            }

            for global_symbol in self.builtin_symbols.keys() {
                globals.insert(global_symbol.to_owned());
            }

            Some(globals)
        };

        module
            .lint(globals.as_ref())
            .into_iter()
            .map(EvalMessage::from)
    }
    pub(crate) fn file_with_contents(
        &self,
        filename: &str,
        content: String,
    ) -> EvalResult<impl Iterator<Item = EvalMessage>> {
        Self::err(
            filename,
            AstModule::parse(filename, content, &self.dialect)
                .map(|module| self.go(filename, module))
                .map_err(Into::into),
        )
    }

    fn get_repository_mapping(&self, repository: &str) -> Option<HashMap<String, String>> {
        let mut raw_command = Command::new("bazel");
        let mut command = raw_command
            .arg("mod")
            .arg("dump_repo_mapping")
            .arg(repository);
        command = command.current_dir(std::env::current_dir().ok()?);

        let output = command.output().ok()?;
        if !output.status.success() {
            return None;
        }

        let output = String::from_utf8(output.stdout).ok()?;
        let entry = output.lines().nth(0)?;
        Some(
            serde_json::from_str::<serde_json::Value>(entry)
                .ok()?
                .as_object()?
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.to_owned(), s.to_owned())))
                .collect(),
        )
    }

    fn get_repository_for_path<'a>(&'a self, path: &'a Path) -> Option<(Cow<'a, str>, &'a Path)> {
        self.external_output_base
            .as_ref()
            .and_then(|external_output_base| path.strip_prefix(external_output_base).ok())
            .and_then(|path| {
                let mut path_components = path.components();

                let repository_name = path_components.next()?.as_os_str().to_string_lossy();
                let repository_path = path_components.as_path();

                Some((repository_name, repository_path))
            })
    }

    fn get_repository_path(&self, repository_name: &str) -> Option<PathBuf> {
        self.external_output_base
            .as_ref()
            .map(|external_output_base| external_output_base.join(repository_name))
    }

    /// Finds the directory that is the root of a package, given a label
    fn resolve_folder(
        &self,
        label: &Label,
        current_file: &LspUrl,
        workspace_root: Option<&Path>,
    ) -> anyhow::Result<PathBuf> {
        // Find the root we're resolving from. There's quite a few cases to consider here:
        // - `repository` is empty, and we're resolving from the workspace root.
        // - `repository` is empty, and we're resolving from a known remote repository.
        // - `repository` is not empty, and refers to the current repository (the workspace).
        // - `repository` is not empty, and refers to a known remote repository.
        //
        // Also with all of these cases, we need to consider if we have build system
        // information or not. If not, we can't resolve any remote repositories, and we can't
        // know whether a repository name refers to the workspace or not.
        let resolve_root = match (&label.repo, current_file) {
            // Repository is empty, and we know what file we're resolving from. Use the build
            // system information to check if we're in a known remote repository, and what the
            // root is. Fall back to the `workspace_root` otherwise.
            (None, LspUrl::File(current_file)) => self
                .get_repository_for_path(current_file)
                .and_then(|(repository, _)| self.get_repository_path(&repository).map(Cow::Owned))
                .or(workspace_root.map(Cow::Borrowed)),
            // No repository in the load path, and we don't have build system information, or
            // an `LspUrl` we can't use to check the root. Use the workspace root.
            (None, _) => workspace_root.map(Cow::Borrowed),
            // We have a repository name and build system information. Check if the repository
            // name refers to the workspace, and if so, use the workspace root. If not, check
            // if it refers to a known remote repository, and if so, use that root.
            // Otherwise, fail with an error.
            (Some(repository), current_file) => {
                let canonical_repo_name: Cow<str> = match repository {
                    repo if repo.is_canonical => Cow::Borrowed(repo.name.as_str()),
                    repo => {
                        let current_repo = match current_file {
                            LspUrl::File(current_file) => self
                                .get_repository_for_path(current_file)
                                .map(|(repository, _)| repository),
                            _ => None,
                        }
                        .unwrap_or(Cow::Borrowed(""));
                        self.get_repository_mapping(&current_repo)
                            .and_then(|mut mapping| mapping.remove(&repo.name).map(Cow::Owned))
                            .unwrap_or(Cow::Borrowed(repo.name.as_str()))
                    }
                };
                if canonical_repo_name.is_empty()
                    || matches!(self.workspace_name.as_ref(), Some(name) if name == &canonical_repo_name)
                {
                    workspace_root.map(Cow::Borrowed)
                } else if let Some(remote_repository_root) = self
                    .get_repository_path(&canonical_repo_name)
                    .map(Cow::Owned)
                {
                    Some(remote_repository_root)
                } else {
                    return Err(ResolveLoadError::UnknownRepository(
                        label.clone(),
                        canonical_repo_name.into_owned(),
                    )
                    .into());
                }
            }
        };

        if let Some(package) = &label.package {
            // Resolve from the root of the repository.
            match resolve_root {
                Some(resolve_root) => Ok(resolve_root.join(package)),
                None => Err(ResolveLoadError::MissingWorkspaceRoot(label.clone()).into()),
            }
        } else {
            // If we don't have a package, this is relative to the current file,
            // so resolve relative paths from the current file.
            match current_file {
                LspUrl::File(current_file_path) => {
                    let current_file_dir = current_file_path.parent();
                    match current_file_dir {
                        Some(current_file_dir) => Ok(current_file_dir.to_owned()),
                        None => Err(ResolveLoadError::MissingCurrentFilePath(label.clone()).into()),
                    }
                }
                _ => Err(
                    ResolveLoadError::WrongScheme("file://".to_owned(), current_file.clone())
                        .into(),
                ),
            }
        }
    }

    fn get_repository_names(&self) -> Vec<Cow<str>> {
        match self.get_repository_mapping("") {
            Some(mapping) => mapping.keys().map(|k| Cow::Owned(k.to_owned())).collect(),
            None => {
                let mut names = Vec::new();
                if let Some(workspace_name) = &self.workspace_name {
                    names.push(Cow::Borrowed(workspace_name.as_str()));
                }

                if let Some(external_output_base) = self.external_output_base.as_ref() {
                    // Look for existing folders in `external_output_base`.
                    if let Ok(entries) = std::fs::read_dir(external_output_base) {
                        for entry in entries.flatten() {
                            if let Ok(file_type) = entry.file_type() {
                                if file_type.is_dir() {
                                    if let Some(name) = entry.file_name().to_str() {
                                        names.push(Cow::Owned(name.to_owned()));
                                    }
                                }
                            }
                        }
                    }
                }

                names
            }
        }
    }

    fn get_filesystem_entries(
        &self,
        from: &Path,
        options: &FilesystemCompletionOptions,
    ) -> anyhow::Result<Vec<FilesystemCompletionResult>> {
        let mut results = Vec::new();

        for entry in fs::read_dir(from)? {
            let entry = entry?;
            let path = entry.path();
            // NOTE: Safe to `unwrap()` here, because we know that `path` is a file system path. And
            // since it's an entry in a directory, it must have a file name.
            let file_name = path.file_name().unwrap().to_string_lossy();
            if path.is_dir() && !path.is_symlink() && options.directories {
                results.push(FilesystemCompletionResult {
                    value: file_name.to_string(),
                    kind: FilesystemCompletionResultKind::Directory,
                });
            } else if path.is_file()
                && !path.is_symlink()
                && options.files != FilesystemFileCompletionOptions::None
                && !Self::BUILD_FILE_NAMES.contains(&file_name.as_ref())
            {
                // Check if it's in the list of allowed extensions. If we have a list, and it
                // doesn't contain the extension, or the file has no extension, skip this file.
                if options.files == FilesystemFileCompletionOptions::OnlyLoadable {
                    let extension = path.extension().map(|ext| ext.to_string_lossy());
                    match extension {
                        Some(extension) => {
                            if !Self::LOADABLE_EXTENSIONS.contains(&extension.as_ref()) {
                                continue;
                            }
                        }
                        None => {
                            continue;
                        }
                    }
                }
                results.push(FilesystemCompletionResult {
                    value: file_name.to_string(),
                    kind: FilesystemCompletionResultKind::File,
                });
            }
        }

        Ok(results)
    }

    fn get_target_entries(
        &self,
        from: &str,
        current_file: &LspUrl,
        workspace_root: Option<&Path>,
    ) -> anyhow::Result<Vec<TargetCompletionResult>> {
        // Find the actual folder on disk we're looking at.
        let package_dir =
            self.resolve_folder(&Label::parse(from)?, current_file, workspace_root)?;
        let build_file_exists = Self::BUILD_FILE_NAMES
            .iter()
            .any(|name| package_dir.join(name).is_file());
        if !build_file_exists {
            return Ok(Vec::new());
        }

        let query_dir = if from.is_empty() {
            Some(package_dir.as_ref())
        } else {
            workspace_root
        };

        let current_file_dir = current_file.path().parent().unwrap();
        let visible_from = (|| {
            if Self::BUILD_FILE_NAMES
                .contains(&current_file.path().file_name()?.to_string_lossy().as_ref())
            {
                if let Ok(relative_path) = current_file_dir.strip_prefix(workspace_root?) {
                    return Some(format!("//{}:*", relative_path.to_string_lossy()));
                }
            }
            None
        })();

        if let Some(targets) =
            self.query_buildable_targets(from, query_dir, visible_from.as_deref())
        {
            Ok(targets
                .into_iter()
                .map(|(kind, value)| TargetCompletionResult { value, kind })
                .collect())
        } else {
            return Ok(Vec::new());
        }
    }

    fn query_buildable_targets(
        &self,
        module: &str,
        workspace_dir: Option<&Path>,
        visible_from: Option<&str>,
    ) -> Option<Vec<(TargetKind, String)>> {
        let mut raw_command = Command::new("bazel");
        let mut command = raw_command.arg("query").arg("--output=label_kind");
        if let Some(visible_from) = visible_from {
            command = command.arg(format!("visible({visible_from}, {module}:*)"));
        } else {
            command = command.arg(format!("{module}:*"));
        }
        if let Some(workspace_dir) = workspace_dir {
            command = command.current_dir(workspace_dir);
        }

        let output = command.output().ok()?;
        if !output.status.success() {
            return None;
        }

        let output = String::from_utf8(output.stdout).ok()?;
        Some(
            output
                .lines()
                .filter_map(|line| {
                    let (kind, label) = line.rsplit_once(' ')?;
                    Some((TargetKind::parse(kind), label.split_once(':')?.1.to_owned()))
                })
                .collect(),
        )
    }
}

impl LspContext for BazelContext {
    fn parse_file_with_contents(&self, uri: &LspUrl, content: String) -> LspEvalResult {
        match uri {
            LspUrl::File(uri) => {
                let EvalResult { messages, ast } =
                    self.file_with_contents(&uri.to_string_lossy(), content);
                LspEvalResult {
                    diagnostics: messages.map(eval_message_to_lsp_diagnostic).collect(),
                    ast,
                }
            }
            _ => LspEvalResult::default(),
        }
    }

    fn resolve_load(
        &self,
        path: &str,
        current_file: &LspUrl,
        workspace_root: Option<&std::path::Path>,
    ) -> anyhow::Result<LspUrl> {
        let label = Label::parse(path)?;

        let folder = self.resolve_folder(&label, current_file, workspace_root)?;

        // Try the presumed filename first, and check if it exists.
        let presumed_path = folder.join(label.name);
        if presumed_path.exists() && !presumed_path.is_dir() {
            return Ok(Url::from_file_path(presumed_path).unwrap().try_into()?);
        }

        // If the presumed filename doesn't exist, try to find a build file from the build system
        // and use that instead.
        for build_file_name in Self::BUILD_FILE_NAMES {
            let path = folder.join(build_file_name);
            if path.exists() {
                return Ok(Url::from_file_path(path).unwrap().try_into()?);
            }
        }

        Err(ResolveLoadError::TargetNotFound(path.to_owned()).into())
    }

    fn render_as_load(
        &self,
        target: &LspUrl,
        current_file: &LspUrl,
        workspace_root: Option<&Path>,
    ) -> anyhow::Result<String> {
        match (target, current_file) {
            // Check whether the target and the current file are in the same package.
            (LspUrl::File(target_path), LspUrl::File(current_file_path)) if matches!((target_path.parent(), current_file_path.parent()), (Some(a), Some(b)) if a == b) =>
            {
                // Then just return a relative path.
                let target_filename = target_path.file_name();
                match target_filename {
                    Some(filename) => Ok(format!(":{}", filename.to_string_lossy())),
                    None => Err(RenderLoadError::MissingTargetFilename(target_path.clone()).into()),
                }
            }
            (LspUrl::File(target_path), _) => {
                // Try to find a repository that contains the target, as well as the path to the
                // target relative to the repository root. If we can't find a repository, we'll
                // try to resolve the target relative to the workspace root. If we don't have a
                // workspace root, we'll just use the target path as-is.
                let (repository, target_path) = &self
                    .get_repository_for_path(target_path)
                    .map(|(repository, target_path)| (Some(repository), target_path))
                    .or_else(|| {
                        workspace_root
                            .and_then(|root| target_path.strip_prefix(root).ok())
                            .map(|path| (None, path))
                    })
                    .unwrap_or((None, target_path));

                let target_filename = target_path.file_name();
                match target_filename {
                    Some(filename) => Ok(format!(
                        "@{}//{}:{}",
                        repository.as_ref().unwrap_or(&Cow::Borrowed("")),
                        target_path
                            .parent()
                            .map(|path| path.to_string_lossy())
                            .unwrap_or_default(),
                        filename.to_string_lossy()
                    )),
                    None => Err(
                        RenderLoadError::MissingTargetFilename(target_path.to_path_buf()).into(),
                    ),
                }
            }
            _ => Err(RenderLoadError::WrongScheme(
                "file://".to_owned(),
                target.clone(),
                current_file.clone(),
            )
            .into()),
        }
    }

    fn resolve_string_literal(
        &self,
        literal: &str,
        current_file: &LspUrl,
        workspace_root: Option<&Path>,
    ) -> anyhow::Result<Option<StringLiteralResult>> {
        self.resolve_load(literal, current_file, workspace_root)
            .map(|url| {
                let original_target_name = Path::new(literal).file_name();
                let path_file_name = url.path().file_name();
                let same_filename = original_target_name == path_file_name;

                Some(StringLiteralResult {
                    url: url.clone(),
                    // If the target name is the same as the original target name, we don't need to
                    // do anything. Otherwise, we need to find the function call in the target file
                    // that has a `name` parameter with the same value as the original target name.
                    location_finder: if same_filename {
                        None
                    } else {
                        match Label::parse(literal) {
                            Err(_) => None,
                            Ok(label) => Some(Box::new(move |ast| {
                                Ok(ast.find_function_call_with_name(&label.name))
                            })),
                        }
                    },
                })
            })
    }

    fn get_load_contents(&self, uri: &LspUrl) -> anyhow::Result<Option<String>> {
        match uri {
            LspUrl::File(path) => match path.is_absolute() {
                true => match fs::read_to_string(path) {
                    Ok(contents) => Ok(Some(contents)),
                    Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
                    Err(e) => Err(e.into()),
                },
                false => Err(ContextError::NotAbsolute(uri.clone()).into()),
            },
            LspUrl::Starlark(_) => Ok(self.builtin_docs.get(uri).cloned()),
            _ => Err(ContextError::WrongScheme("file://".to_owned(), uri.clone()).into()),
        }
    }

    fn get_environment(&self, _uri: &LspUrl) -> DocModule {
        DocModule::default()
    }

    fn get_url_for_global_symbol(
        &self,
        _current_file: &LspUrl,
        symbol: &str,
    ) -> anyhow::Result<Option<LspUrl>> {
        Ok(self.builtin_symbols.get(symbol).cloned())
    }

    fn get_string_completion_options(
        &self,
        document_uri: &LspUrl,
        kind: StringCompletionType,
        current_value: &str,
        cursor_offset: usize,
        workspace_root: Option<&Path>,
    ) -> anyhow::Result<Vec<StringCompletionResult>> {
        let before_cursor = &current_value[..cursor_offset];
        let offer_repository_names = before_cursor.starts_with('@') && !before_cursor.contains('/');

        let mut names = if offer_repository_names {
            self.get_repository_names()
                .into_iter()
                .map(|name| {
                    let name_with_at = format!("@{}", name);
                    let text_edit = StringCompletionTextEdit {
                        begin: 0,
                        end: current_value
                            .find("//")
                            .map(|x| x + 2)
                            .unwrap_or(current_value.len()),
                        text: format!("{}//", &name_with_at),
                    };
                    StringCompletionResult {
                        label: name_with_at,
                        text_edit,
                        additional_text_edits: None,
                        kind: CompletionItemKind::MODULE,
                        detail: None,
                        trigger_another_completion: true,
                    }
                })
                .collect()
        } else {
            vec![]
        };

        // "fo" -> target ":foo" (if it's not source file)
        //      -> directory "foo/" (if `kind` is `String`)
        //      -> file "foo" (if `kind` is `String`)
        //      -> file ":foo.bzl" (if `kind` is `LoadPath`)
        // "foo/ba" -> target ":foo/bar" (if it's not source file)
        //          -> directory "foo/bar/" (if `kind` is `String`)
        //          -> file "foo/bar" (if `kind` is `String`)
        // ":fo" -> target ":foo" (if it's not source file)
        //       -> file ":foo.bzl" (if `kind` is `LoadPath`)
        // "@fo" -> repo "@foo//"
        // "@foo/" -> (None)
        // "@foo//ba" -> target "@foo//:bar"
        //            -> directory "@foo//bar/"
        //            -> file "@foo//:bar.bzl" (if `kind` is `LoadPath`)
        // "@foo//:ba" -> target "@foo//:bar"
        //             -> file "@foo//:bar.bzl" (if `kind` is `LoadPath`)
        // "@foo//bar/ba" -> target "@foo//bar:baz"
        //                -> directory "@foo//bar/baz/"
        //                -> file "@foo//bar:baz.bzl" (if `kind` is `LoadPath`)
        // "@foo//bar:ba" -> target "@foo//bar:baz"
        //                -> file "@foo//bar:baz.bzl" (if `kind` is `LoadPath`)
        let cursor_in_repo = before_cursor.starts_with('@') && !before_cursor.contains("//");
        let source_file_like = !before_cursor.starts_with('@')
            && !before_cursor.starts_with('/')
            && !before_cursor.starts_with(':');
        let complete_directories = !cursor_in_repo
            && !before_cursor.contains(':')
            && !(kind == StringCompletionType::LoadPath && source_file_like);
        let complete_filenames = !cursor_in_repo
            && (kind == StringCompletionType::LoadPath || source_file_like)
            && !(kind == StringCompletionType::LoadPath
                && source_file_like
                && before_cursor.contains('/'));
        let complete_targets = !cursor_in_repo && kind == StringCompletionType::String;

        let root_package = if source_file_like {
            ""
        } else {
            before_cursor
                .rsplit_once(':')
                .map(|(package, _)| package)
                .or_else(|| {
                    let pos = before_cursor.rfind('/')?;
                    let package = &before_cursor[..pos + 1];
                    if package.ends_with("//") {
                        Some(package)
                    } else {
                        Some(&before_cursor[..pos])
                    }
                })
                .unwrap_or("")
        };
        if complete_directories || complete_filenames {
            let render_base = if source_file_like {
                before_cursor
                    .rsplit_once('/')
                    .map(|(dir, _)| dir)
                    .unwrap_or("")
            } else {
                root_package
            };
            let root_path = if source_file_like {
                document_uri.path().parent().unwrap().join(render_base)
            } else {
                self.resolve_folder(&Label::parse(root_package)?, document_uri, workspace_root)?
            };
            let filesystem_entries = self
                .get_filesystem_entries(
                    &root_path,
                    &FilesystemCompletionOptions {
                        directories: complete_directories,
                        files: match (&kind, complete_filenames) {
                            (StringCompletionType::LoadPath, true) => {
                                FilesystemFileCompletionOptions::OnlyLoadable
                            }
                            (StringCompletionType::String, true) => {
                                FilesystemFileCompletionOptions::All
                            }
                            (_, false) => FilesystemFileCompletionOptions::None,
                        },
                    },
                )
                .unwrap_or_default();
            names.extend(filesystem_entries.into_iter().map(|entry| {
                let with_colon_or_slash = before_cursor[render_base.len()..].starts_with(':')
                    || before_cursor[render_base.len()..].starts_with('/');
                let drop_colon_or_slash = kind == StringCompletionType::LoadPath
                    && entry.kind == FilesystemCompletionResultKind::File
                    && with_colon_or_slash;
                let text = match (&entry.kind, &kind) {
                    (FilesystemCompletionResultKind::Directory, _) => format!("{}/", entry.value),
                    (_, StringCompletionType::String) => format!("{}", entry.value),
                    (_, StringCompletionType::LoadPath) => format!(":{}", entry.value),
                };
                let text_edit = StringCompletionTextEdit {
                    begin: render_base.chars().count() + if with_colon_or_slash { 1 } else { 0 },
                    end: match &entry.kind {
                        FilesystemCompletionResultKind::Directory => current_value[cursor_offset..]
                            .find("/")
                            .map(|x| before_cursor.chars().count() + x + 1)
                            .unwrap_or(current_value.chars().count()),
                        FilesystemCompletionResultKind::File => current_value.chars().count(),
                    },
                    text,
                };
                let additional_text_edits = if drop_colon_or_slash {
                    Some(vec![StringCompletionTextEdit {
                        begin: render_base.chars().count(),
                        end: render_base.chars().count() + 1,
                        text: String::from(""),
                    }])
                } else {
                    None
                };
                StringCompletionResult {
                    label: entry.value,
                    text_edit,
                    additional_text_edits,
                    kind: match &entry.kind {
                        FilesystemCompletionResultKind::Directory => CompletionItemKind::FOLDER,
                        FilesystemCompletionResultKind::File => CompletionItemKind::FILE,
                    },
                    detail: None,
                    trigger_another_completion: match &entry.kind {
                        FilesystemCompletionResultKind::Directory => true,
                        FilesystemCompletionResultKind::File => false,
                    },
                }
            }));
        }
        if complete_targets {
            let target_entries =
                self.get_target_entries(&root_package, document_uri, workspace_root)?;
            names.extend(
                target_entries
                    .into_iter()
                    // Filter out relative source files as they were already handled by filesystem completion
                    .filter(|entry| {
                        !((source_file_like || before_cursor.starts_with(':'))
                            && entry.kind == TargetKind::SourceFile)
                    })
                    .map(|entry| {
                        let drop_colon_or_slash = before_cursor[root_package.len()..]
                            .starts_with(':')
                            || before_cursor[root_package.len()..].starts_with('/');
                        let text_edit = StringCompletionTextEdit {
                            begin: root_package.chars().count()
                                + if drop_colon_or_slash { 1 } else { 0 },
                            end: current_value.chars().count(),
                            text: format!(":{}", entry.value),
                        };
                        let additional_text_edits = if drop_colon_or_slash {
                            Some(vec![StringCompletionTextEdit {
                                begin: root_package.chars().count(),
                                end: root_package.chars().count() + 1,
                                text: String::from(""),
                            }])
                        } else {
                            None
                        };
                        StringCompletionResult {
                            label: entry.value,
                            text_edit,
                            additional_text_edits,
                            kind: match entry.kind {
                                TargetKind::SourceFile => CompletionItemKind::FILE,
                                TargetKind::GeneratedFile => CompletionItemKind::REFERENCE,
                                TargetKind::Rule(_) => CompletionItemKind::FIELD,
                                TargetKind::Unknown(_) => CompletionItemKind::PROPERTY,
                            },
                            detail: Some(entry.kind.to_string()),
                            trigger_another_completion: false,
                        }
                    }),
            );
        }

        Ok(names)
    }
}
