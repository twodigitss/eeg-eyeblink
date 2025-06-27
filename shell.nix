{ pkgs ? import <nixpkgs> {} }:

let
  # Definir Python con los paquetes necesarios
  pythonEnv = pkgs.python312.withPackages (ps: with ps; [
    pybluez

    # Análisis de datos básico
      # Matematico
      numpy
      pandas
      scipy
      
      # Visualización
      matplotlib
      seaborn
      plotly
    
  ]);
  
in pkgs.mkShell {
  buildInputs = with pkgs; [
    pythonEnv
  ];
  
  shellHook = ''
    # Configurar variables de entorno útiles
    export PYTHONPATH="${pythonEnv}/lib/python3.11/site-packages:$PYTHONPATH"
    export JUPYTER_CONFIG_DIR="$PWD/.jupyter"
    export JUPYTER_DATA_DIR="$PWD/.local/share/jupyter"
    
    # Configuración para matplotlib en sistemas sin display
    export MPLBACKEND=Agg
    export PS1="[🐬 \W] $ "
  '';
  
  # Variables de entorno adicionales
  PYTHONUNBUFFERED = "1";
  PYTHONDONTWRITEBYTECODE = "1";
}
