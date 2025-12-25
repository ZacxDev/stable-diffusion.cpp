{
  description = "Diffusion model (SD, Flux, Wan, etc.) inference in pure C/C++";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Default pkgs (no unfree)
        pkgs = nixpkgs.legacyPackages.${system};

        # pkgs with unfree allowed (for CUDA)
        pkgs-unfree = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Common build inputs for all variants
        commonBuildInputs = with pkgs; [
          cmake
          pkg-config
        ];

        # Development tools
        devTools = with pkgs; [
          clang-tools  # for clang-format and clang-tidy
          git
          gdb
          valgrind  # for memory leak detection
          imagemagick  # for image inspection (identify, convert)
        ];

        # Python environment for benchmarking and charts
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          matplotlib
          numpy
        ]);

        # Fetch ggml separately since it's a git submodule
        ggml-src = pkgs.fetchFromGitHub {
          owner = "ggml-org";
          repo = "ggml";
          rev = "3e9f2ba3b934c20b26873b3c60dbf41b116978ff";
          hash = "sha256-+gRMUTpi1dMMG2NNMkqnoQ98I4OMs4L6uaP1Q0N51sc=";
        };

        # Create a base package builder function
        mkStableDiffusion = { pkgsToUse ? pkgs
                            , enableCuda ? false
                            , enableVulkan ? false
                            , enableMetal ? false
                            , enableOpenCL ? false
                            , enableSycl ? false
                            , enableHipblas ? false
                            , enableMusa ? false
                            , buildExamples ? true
                            , buildSharedLibs ? false
                            }:
          pkgsToUse.stdenv.mkDerivation {
            pname = "stable-diffusion-cpp";
            version = "unstable-${self.lastModifiedDate or "unknown"}";

            # Use builtins.path to include everything including submodules
            # Filter out build artifacts and git metadata
            src = builtins.path {
              path = ./.;
              name = "source";
              filter = path: type:
                let
                  baseName = baseNameOf path;
                  relPath = pkgsToUse.lib.removePrefix (toString ./. + "/") (toString path);
                in
                # Exclude build directories, git metadata, and common artifacts
                baseName != ".git" &&
                !pkgsToUse.lib.hasPrefix "build" relPath &&
                !pkgsToUse.lib.hasPrefix "cmake-build" relPath &&
                !pkgsToUse.lib.hasPrefix ".serena" relPath &&
                baseName != "result" &&
                baseName != "result-bin";
            };

            nativeBuildInputs = with pkgsToUse; [
              cmake
              pkg-config
              rsync  # for copying ggml submodule
            ] ++ pkgsToUse.lib.optionals enableCuda [
              # addDriverRunpath patches binaries to find NVIDIA driver libraries
              pkgsToUse.autoAddDriverRunpath
            ];

            # Copy ggml submodule content after unpacking
            postUnpack = ''
              # Copy ggml from fetched source (git submodules aren't included in flake sources)
              echo "Copying ggml submodule from fetched source"
              mkdir -p source/ggml
              cp -r ${ggml-src}/* source/ggml/
              chmod -R u+w source/ggml
            '';

            buildInputs = with pkgsToUse; [
              # Always needed
            ] ++ pkgsToUse.lib.optionals enableCuda [
              cudaPackages.cudatoolkit
            ] ++ pkgsToUse.lib.optionals enableVulkan [
              vulkan-headers
              vulkan-loader
            ] ++ pkgsToUse.lib.optionals enableOpenCL [
              ocl-icd
              opencl-headers
            ] ++ pkgsToUse.lib.optionals enableMetal [
              # Metal is macOS-only, handled by stdenv
            ] ++ pkgsToUse.lib.optionals enableSycl [
              # SYCL support would need oneAPI
            ];

            cmakeFlags = [
              "-DCMAKE_BUILD_TYPE=Release"
              "-DSD_BUILD_EXAMPLES=${if buildExamples then "ON" else "OFF"}"
              "-DSD_BUILD_SHARED_LIBS=${if buildSharedLibs then "ON" else "OFF"}"
              "-DSD_CUDA=${if enableCuda then "ON" else "OFF"}"
              "-DSD_VULKAN=${if enableVulkan then "ON" else "OFF"}"
              "-DSD_METAL=${if enableMetal then "ON" else "OFF"}"
              "-DSD_OPENCL=${if enableOpenCL then "ON" else "OFF"}"
              "-DSD_SYCL=${if enableSycl then "ON" else "OFF"}"
              "-DSD_HIPBLAS=${if enableHipblas then "ON" else "OFF"}"
              "-DSD_MUSA=${if enableMusa then "ON" else "OFF"}"
              "-DSD_SDTENSORS_SUPPORT=ON"  # Enable .sdtensors binary format support
            ];

            # For CUDA builds, patch binaries to find system NVIDIA drivers at /run/opengl-driver/lib
            postFixup = pkgsToUse.lib.optionalString enableCuda ''
              for bin in $out/bin/*; do
                if [ -f "$bin" ] && [ -x "$bin" ]; then
                  echo "Patching RPATH for $bin to include NVIDIA driver path"
                  oldRpath=$(patchelf --print-rpath "$bin" || echo "")
                  patchelf --set-rpath "/run/opengl-driver/lib:$oldRpath" "$bin" || true
                fi
              done
            '';

            meta = with pkgsToUse.lib; {
              description = "Diffusion model (SD, Flux, Wan, etc.) inference in pure C/C++";
              homepage = "https://github.com/leejet/stable-diffusion.cpp";
              license = licenses.mit;
              maintainers = [ ];
              platforms = platforms.unix;
              mainProgram = "sd-cli";
            };
          };

      in
      {
        # Default package (CPU-only build)
        packages.default = mkStableDiffusion {
          buildExamples = true;
        };

        # CPU-only package
        packages.cpu = mkStableDiffusion {
          buildExamples = true;
        };

        # CUDA-enabled package
        packages.cuda = mkStableDiffusion {
          pkgsToUse = pkgs-unfree;
          enableCuda = true;
          buildExamples = true;
        };

        # Vulkan-enabled package
        packages.vulkan = mkStableDiffusion {
          enableVulkan = true;
          buildExamples = true;
        };

        # OpenCL-enabled package
        packages.opencl = mkStableDiffusion {
          enableOpenCL = true;
          buildExamples = true;
        };

        # Shared library package
        packages.shared = mkStableDiffusion {
          buildSharedLibs = true;
          buildExamples = true;
        };

        # Development shell
        devShells.default = pkgs.mkShell {
          name = "stable-diffusion-cpp-dev";

          buildInputs = commonBuildInputs ++ devTools ++ (with pkgs; [
            # Optional: CUDA support for development
            # cudaPackages.cudatoolkit

            # Optional: Vulkan support for development
            vulkan-headers
            vulkan-loader
            vulkan-tools

            # Optional: OpenCL support for development
            ocl-icd
            opencl-headers

            # Python environment for benchmarking
            pythonEnv
          ]);

          shellHook = ''
            echo "🎨 stable-diffusion.cpp development environment"
            echo ""
            echo "Available commands:"
            echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release"
            echo "  cmake --build build --config Release"
            echo "  ./format-code.sh    # Format code with clang-format"
            echo ""
            echo "Build variants:"
            echo "  -DSD_CUDA=ON        # Enable CUDA backend"
            echo "  -DSD_VULKAN=ON      # Enable Vulkan backend"
            echo "  -DSD_OPENCL=ON      # Enable OpenCL backend"
            echo "  -DSD_METAL=ON       # Enable Metal backend (macOS)"
            echo ""
            echo "Git submodules:"
            if [ ! -d "ggml/.git" ]; then
              echo "  ⚠️  ggml submodule not initialized!"
              echo "  Run: git submodule update --init --recursive"
            else
              echo "  ✓ ggml submodule initialized"
            fi
            echo ""
          '';

          # Set C++ standard
          NIX_CFLAGS_COMPILE = "-std=c++17";
        };

        # Additional dev shell with CUDA
        devShells.cuda = pkgs-unfree.mkShell {
          name = "stable-diffusion-cpp-cuda-dev";

          buildInputs = (with pkgs-unfree; [
            cmake
            pkg-config
            clang-tools
            git
            gdb
            cudaPackages.cudatoolkit
            vulkan-headers
            vulkan-loader
          ]) ++ [
            # Add valgrind and Python from regular pkgs (not unfree)
            pkgs.valgrind
            pythonEnv
          ];

          shellHook = ''
            # Add NVIDIA driver libraries to LD_LIBRARY_PATH so CUDA can find real drivers
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH

            echo "🎨 stable-diffusion.cpp CUDA development environment"
            echo ""
            echo "CUDA toolkit available"
            echo "NVIDIA driver path: /run/opengl-driver/lib"
            echo "Build with: cmake -B build -DSD_CUDA=ON"
            echo ""
            echo "Git submodules:"
            if [ ! -d "ggml/.git" ]; then
              echo "  ⚠️  ggml submodule not initialized!"
              echo "  Run: git submodule update --init --recursive"
            else
              echo "  ✓ ggml submodule initialized"
            fi
            echo ""
          '';

          NIX_CFLAGS_COMPILE = "-std=c++17";
        };

        # App for easy running
        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/sd-cli";
        };

        apps.sd = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/sd-cli";
        };
      }
    );
}
