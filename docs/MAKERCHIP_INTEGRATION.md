# Makerchip browser integration

`Maker → Verilog → Author → Edit in Makerchip IDE` uses Makerchip's
[supported `IdePlugin` v2 browser API](https://makerchip.com/public/docs/plugin_api/).
It does not use the deprecated
`makerchip-app` Python package or Makerchip's retired public-project desktop
service.

## User flow

1. eSim materializes the in-memory Author design so Makerchip receives the
   current text.
2. For a `.v` / `.sv` design, the existing choice remains:
   - **Yes** generates and opens a Makerchip-ready `.tlv` wrapper beside the
     design.
   - **No** opens the Verilog file unchanged.
3. eSim starts a session on `127.0.0.1` and opens its URL in the default
   browser. The page imports
   `https://makerchip.com/dist/makerchip-plugin.js`, then creates an editable
   IDE with the file contents.
4. Makerchip's `onCodeChange()` callback is debounced and saved to the exact
   file that was opened. If this is the Author design, `DesignBus` reports the
   external change through its existing non-modal **Reload / Keep mine** bar.

Makerchip remains a third-party editor/simulator. Browser compilation does not
silently advance eSim to Verify or Convert, import a VCD, or build an ngspice
model.

## Conflict and failure behavior

Every read returns a SHA-256 revision of the file bytes. A browser save must
name the revision it edited. If eSim or another editor changed the file in the
meantime, the bridge returns `409 Conflict` and does not write. The browser
offers two explicit choices: reload the latest file or keep the browser edit.

The bridge reports browser/network initialization failures in its status bar.
eSim reports a missing design or failure to open the system browser before
returning control to the user.

## Security boundary

- The HTTP server binds to `127.0.0.1`, never a LAN or wildcard address.
- Every route contains a new 256-bit random session token.
- Writes accept JSON only, are capped at 10 MiB, and can reach only the single
  absolute file selected by eSim when the session was created.
- Responses disable caching and referrer transmission; no CORS permission is
  granted.
- Closing the Author widget shuts down the bridge. Reopening Makerchip replaces
  the previous session and invalidates its token.

The production plugin URL is intentional. Beta documentation and examples may
describe unreleased API additions, while the stable entry module pins its own
Makerchip IDE version.
