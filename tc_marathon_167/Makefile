.PHONY: build test bundle bench clean
PYTHON ?= python3
CMAKE ?= cmake
SEEDS ?= 1:10

build:
	$(CMAKE) -S . -B build -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build build -j

test: build
	ctest --test-dir build --output-on-failure
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

# Single-file submission: dist/WordGame.cpp and dist/WordGame.zip.
bundle:
	$(PYTHON) scripts/bundle.py

bench: build
	$(PYTHON) scripts/run_tester.py --exe build/WordGame --seeds $(SEEDS)

clean:
	$(CMAKE) -E rm -rf build dist
