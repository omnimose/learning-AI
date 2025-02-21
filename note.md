1. To display LaTex in VS Code Jyputer Notebook markdown:
https://stackoverflow.com/questions/62879232/how-do-i-use-latex-in-a-jupyter-notebook-inside-visual-studio-code

You can now with the latest version of VS Code. The feature is still in preview (as of writing) so you have to enable it in settings.json by adding "notebook.experimental.useMarkdownRenderer": true

View -> Command Palette -> Preferences: Open User Settings (JSON)

My user settings:

{
    "go.toolsManagement.autoUpdate": true,
    "window.zoomLevel": 1,
    "github.copilot.enable": {
        "*": true,
        "plaintext": false,
        "markdown": true,
        "scminput": false
    },
    "editor.inlineSuggest.enabled": true,
    "editor.inlineSuggest.showToolbar": "always",
    "workbench.editor.enablePreview": false,
    "notebook.output.textLineLimit": 100,
    "notebook.output.scrolling": true,
    "notebook.experimental.useMarkdownRenderer": true,
    "editor.minimap.enabled": false,
    "remote.SSH.remotePlatform": {
        "172.203.231.10": "linux"
    },
    "cmake.showOptionsMovedNotification": false,
    "git.autofetch": true,
    "editor.renderWhitespace": "all",
    "diffEditor.ignoreTrimWhitespace": false,
    "rust-analyzer.inlayHints.closureReturnTypeHints.enable": "always",
    "rust-analyzer.inlayHints.closureCaptureHints.enable": true,
    "rust-analyzer.inlayHints.typeHints.enable": false,
    "editor.inlayHints.enabled": "offUnlessPressed"
}

Seems inline LaTex is not supported yet.

- Seems the Inline formula: Use \( ... \) used by ChatGPT is not supported; use $...$ instead.
- Seems \[...\] does not work also for block formula; use $$ ... $$ instead.