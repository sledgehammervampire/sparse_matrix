let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs { };
  naersk = pkgs.callPackage sources.naersk { };
  sparseMatrix = naersk.buildPackage {
    root = ./.;
    doCheck = true;
  };
in pkgs.mkShell {
  buildInputs = [ sparseMatrix ] ++ (with pkgs; [ cargo cargo-edit ]);
}
