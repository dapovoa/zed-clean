# Zed Settings Documentation

## 7. Local: Terminal de Execução (clean.terminal.*)

The `clean.terminal.*` namespace provides theme customization for terminal-related UI elements:

- `background`: Fundo da consola real.
- `foreground`: Cor do texto da consola.
- `selection`: Cor de seleção na consola.

Example configuration:

```json
{
  "clean.theme_overrides": {
    "clean.terminal.background": "#0d1117",
    "clean.terminal.foreground": "#f0f6fc",
    "clean.terminal.selection": "#3b82f655"
  }
}
```