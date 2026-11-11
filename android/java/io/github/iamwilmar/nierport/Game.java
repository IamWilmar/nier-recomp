package io.github.iamwilmar.nierport;

// What identifies this game in the user's files: the rest of the start screen is the same in every port.
// Lo que distingue a este juego en los archivos del usuario: el resto de la pantalla de inicio es igual en cada port.
final class Game {
    static final String NAME = "NieR:Automata The End of YoRHa Edition";
    static final long APP_ID = 0x0100B8E016F76000L;
    // The port only works with this exact version (its addresses and patches are those of this main): recognised by the
    // build ID of the ExeFS main (bytes 0x40-0x53 of the NSO header).
    static final String VERSION = "1.0.2";
    static final String BUILD_ID = "992787e2b54259948b844059a8ab21ad619d44de";
    static final String ROMFS_MARKER = "data000.cpk";  // relative to the RomFS root
    static final String UPDATE_MARKER = null;  // only present with the update applied (null: not checked)
    static final String DLC_MARKER = null;  // in the root of each DLC RomFS (null: game without DLC)

    private Game() {}
}
