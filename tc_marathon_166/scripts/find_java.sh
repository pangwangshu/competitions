find_java() {
  # macOS ships a /usr/bin/java stub that exists even with no JDK installed,
  # so check it actually runs rather than just that the binary is present.
  if command -v java >/dev/null 2>&1 && java -version >/dev/null 2>&1; then
    echo "java"
    return
  fi
  for prefix in openjdk openjdk@21 openjdk@17 openjdk@11; do
    if command -v brew >/dev/null 2>&1; then
      candidate="$(brew --prefix "$prefix" 2>/dev/null)/bin/java"
      if [ -x "$candidate" ]; then
        echo "$candidate"
        return
      fi
    fi
  done
  echo "ERROR: no Java runtime found (tried PATH and brew openjdk)" >&2
  exit 1
}
