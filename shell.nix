let
  sources = import ./nix/sources.nix;
  nixpkgs-mozilla = import sources.nixpkgs-mozilla;
  pkgs = import sources.nixpkgs {
    overlays = [
      nixpkgs-mozilla
      (self: super: {
        rustc = self.latest.rustChannels.stable.rust;
        # don't use cargo from overlay since broken
      })
    ];
  };
  naersk = pkgs.callPackage sources.naersk { };
  rust = pkgs.latest.rustChannels.stable.rust;
  sparseMatrix = naersk.buildPackage {
    root = ./.;
    buildInputs = with pkgs; [];
  };
in pkgs.mkShell {
  buildInputs = [ sparseMatrix rust ]
    ++ (with pkgs; [ cargo-edit ]);
}
