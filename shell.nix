{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    cargo-edit
    cargo-flamegraph
    lld_11
  ];
}
