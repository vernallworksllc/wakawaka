# WakaWaka

WakaWaka (`.wk`) is a modern, multi-paradigm, AI-native programming language from OpenCan.ai that fuses Rust-style safety, Go-style concurrency, Python-style expressiveness, and Zig-style compile-time evaluation. It runs interpreted via `waka run` or transpiles to C99 for native binaries, and ships batteries-included: a formatter, type checker, REPL, web server, and first-class tensors, autodiff, ML pipelines, and model I/O. Safety is the default — Option/Result types eliminate null crashes, while shell and SQL literals auto-escape interpolated values to prevent injection.

## Language Basics & Syntax

### Variables & Constants

Declare immutable bindings with `let`, mutable ones with `var`, and compile-time constants with `const`. Types are inferred but can be annotated explicitly anywhere.

```
let name = "Wakawaka";           // immutable, inferred as str
var count = 0;                   // mutable int
const VERSION = "1.0.0";         // compile-time constant
let x: int = 42;                 // explicit annotation
let maybe: ?str = none;          // optional type
```

**Use case:** Lock down values that should never change while keeping mutable counters and accumulators clearly marked.

### F-String Interpolation & Format Specifiers

F-strings embed arbitrary expressions inside `{}` and support format specifiers for precision, hex, binary, and octal output.

```
let pi = 3.14159;
println(f"x={x}, y={y:.2f}, flag={flag}");
println(f"{pi:.2f}");    // "3.14"
println(f"{255:x}");     // "ff"
println(f"{10:b}");      // "1010"
```

**Use case:** Build readable, formatted log lines and reports without manual string concatenation.

### Collections: Arrays, Maps & Tuples

Dynamic arrays, string-keyed maps, and fixed tuples are built in, with rich methods and Python-style slice syntax.

```
var nums = [1, 2, 3, 4, 5];
nums.push(6);
let scores = {"Alice": 95, "Bob": 87};
for name in scores.keys() {
    println(f"  {name}: {scores.get(name)}");
}
nums[1:4];      // [2, 3, 4]  (slice)
```

**Use case:** Hold tabular lookups (maps), ordered data (arrays), and grouped return values (tuples) without external libraries.

### Control Flow

WakaWaka offers `if/else`, `while`, `do-while`, range and C-style `for` loops, plus labeled `break` for escaping nested loops.

```
for n in 1..=5 { print(f"{n} "); }
for (let i = 0; i < 5; i += 1) { print(f"{i} "); }
outer:
for i in 0..5 {
    for j in 0..5 {
        if i + j >= 6 { break outer; }
    }
}
```

**Use case:** Iterate ranges and collections naturally, then cleanly bail out of deeply nested search loops.

## Functions & Functional Programming

### Functions with Defaults & Multiple Returns

Functions support typed parameters, default arguments (callable by name), variadics, and tuple-style multiple return values.

```
fn greet(name: str, greeting: str = "Hello") -> str {
    return f"{greeting}, {name}!";
}
greet("Bob", greeting: "Hi");

fn divmod(a: int, b: int) -> (int, int) {
    return a / b, a % b;
}
let q, r = divmod(17, 5);   // q=3, r=2
```

**Use case:** Define flexible APIs with optional configuration and return several results without wrapper structs.

### Closures & Higher-Order Functions

Anonymous functions use the `|args| body` syntax, capture their environment, and compose freely into higher-order functions.

```
fn make_adder(n: int) -> any { return |x| x + n; }
fn compose(f: any, g: any) -> any { return |x| f(g(x)); }

let add5 = make_adder(5);
let mul3 = |x| x * 3;
let add5_then_mul3 = compose(mul3, add5);
println(add5_then_mul3(10));   // 45
```

**Use case:** Build reusable transformation pipelines and callback-driven logic such as event handlers or comparators.

### Pipe Operator

The `|>` operator threads a value through a chain of functions, passing the left side as the first argument to the right — ideal for left-to-right data flow.

```
let result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    |> filter(|n| n % 2 == 0)
    |> map(|n| n * n)
    |> sum;
println(f"Sum of squares of evens: {result}");   // 220
```

**Use case:** Express multi-step data transformations as a readable, top-to-bottom pipeline instead of nested calls.

## Types, Objects & Generics

### Structs & Impl Blocks

Structs hold plain data; methods are attached separately via `impl` blocks, keeping data and behavior cleanly separable.

```
struct Point { x: float, y: float }

impl Point {
    fn distance(self, other: Point) -> float {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        return math.sqrt(dx*dx + dy*dy);
    }
}
```

**Use case:** Model lightweight value types like coordinates or records and attach geometry/utility methods to them.

### Classes, Inheritance & Interfaces

Full OOP: classes with constructors, `abstract` base classes, `extends` inheritance with `override`, and `interface` contracts.

```
interface Drawable {
    fn draw(self);
    fn area(self) -> float;
}

abstract class Shape {
    var color: str;
    abstract fn area(self) -> float;
}

class Circle extends Shape {
    var radius: float;
    fn new(r: float, color: str) -> Circle {
        return Circle { radius: r, color: color };
    }
    override fn area(self) -> float {
        return math.PI * self.radius * self.radius;
    }
}
```

**Use case:** Build polymorphic hierarchies (e.g. a list of shapes) where each subtype computes its own area.

### Operator Overloading

Classes can define `operator+`, `operator==`, `operator<`, and more to give domain types natural arithmetic and comparison semantics.

```
class Vec2 {
    var x: float;
    var y: float;
    fn operator+(self, other: Vec2) -> Vec2 {
        return Vec2 { x: self.x + other.x, y: self.y + other.y };
    }
}
let c = a + b;    // Vec2 { x: 4.0, y: 6.0 }
```

**Use case:** Make math-heavy types like vectors, matrices, or money behave like built-in numbers.

### Generics

Generic functions and classes parameterize over types with `<T>`, enabling reusable, type-safe containers and algorithms.

```
class Stack<T> {
    var items: []T;
    fn new() -> Stack<T> { return Stack { items: [] }; }
    fn push(var self, item: T) { self.items.push(item); }
    fn pop(var self) -> ?T {
        if self.items.isEmpty() { return none; }
        return some(self.items.pop());
    }
}
```

**Use case:** Implement one stack, queue, or linked-list type that works for any element type without duplication.

### Enums with Associated Data

Enums model fixed sets of variants, optionally carrying typed payloads — a foundation for algebraic data modeling.

```
enum Shape {
    Circle(float),
    Rectangle(float, float),
    Triangle(float, float, float),
}
let s = Shape.Circle(5.0);
let r = Shape.Rectangle(4.0, 6.0);
```

**Use case:** Represent a closed set of states or shapes where each case carries its own relevant data.

## Pattern Matching & Error Handling

### Match Expressions

`match` supports literal, range, or-patterns, tuple/struct destructuring, binding patterns, and `if` guards — and returns a value.

```
let label = match v {
    0        => "zero",
    1        => "one",
    2..=9    => "small",
    10..=99  => "medium",
    _        => "large",
};
```

**Use case:** Classify inputs or dispatch on enum variants exhaustively, with the compiler guarding against missed cases.

### Option Type (no null crashes)

The `?T` option type encodes "value or nothing" with `some(v)` and `none`, eliminating null-pointer errors. Optional chaining (`?.`) and null-coalescing (`??`) make access ergonomic.

```
fn find(items: []str, target: str) -> ?int {
    for i in 0..items.len() {
        if items[i] == target { return some(i); }
    }
    return none;
}
let name = user?.name ?? "Anonymous";
```

**Use case:** Safely represent lookups and optional fields that may be absent, forcing callers to handle the empty case.

### Result Type & the ? Propagation Operator

`Result<T, E>` makes failure explicit via `ok(v)` / `err(e)`. The `?` operator unwraps a result or returns the error early, chaining fallible calls cleanly.

```
fn process(a_str: str, b_str: str) -> Result<str, str> {
    let a = parse_num(a_str)?;
    let b = parse_num(b_str)?;
    let result = safe_divide(a, b)?;
    return ok(f"{a} / {b} = {result:.4f}");
}
```

**Use case:** Write parsing and I/O code where every step can fail, propagating errors without nested checks.

## Concurrency

### Goroutines & Channels

Launch lightweight background threads with `go` and communicate over typed, buffered channels using the `<-` send operator.

```
fn worker(id: int, ch: any) {
    for i in 0..3 {
        ch <- f"worker-{id} item-{i}";
        sleep(0.05);
    }
}
let ch = chan<str>(10);
go worker(1, ch);
go worker(2, ch);
let msg = recv(ch);
```

**Use case:** Implement producer-consumer pipelines and parallel workers that exchange data safely without shared-memory locks.

### Async / Await

Mark functions `async` and `await` their results for non-blocking I/O such as HTTP fetches.

```
async fn fetch_data(url: str) -> Result<str, str> {
    sleep(0.01);
    return ok(f"data from {url}");
}
let responses = await fetch_all(urls);
```

**Use case:** Fetch multiple network resources concurrently without blocking the main thread.

### Actors

The `actor` keyword defines isolated, thread-safe stateful units, instantiated with `spawn`, providing race-free state management.

```
actor Counter {
    var count: int;
    fn new(initial: int) -> Counter { return Counter { count: initial }; }
    fn increment(var self, n: int) { self.count += n; }
    fn get(self) -> int { return self.count; }
}
let c = spawn Counter.new(0);
```

**Use case:** Maintain shared mutable state (counters, caches) accessed from many goroutines without data races.

### Defer for Resource Cleanup

Deferred statements run in reverse (LIFO) order when the enclosing function returns, guaranteeing cleanup even on early exit.

```
fn with_cleanup(name: str) {
    defer println(f"  cleanup: {name} closed");
    defer println(f"  cleanup: {name} flushed");
    println(f"  working with {name}");
}
// runs: flushed, then closed
```

**Use case:** Reliably close files, release locks, or flush buffers no matter how a function returns.

## AI, Tensors & Machine Learning

### Native Tensors

First-class n-dimensional float tensors with factory constructors, element-wise arithmetic, broadcasting, reductions, reshaping, and linear algebra — no external dependencies.

```
let c = tensor.from([1.0, 2.0, 3.0]);
let d = c + tensor.ones([3]);
println(f"sum={c.sum()}, mean={c.mean()}");
let mat = tensor.from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3]);
let A = tensor.from([[1, 2], [3, 4]]);
let C = A.matmul(tensor.from([[5, 6], [7, 8]]));   // [[19,22],[43,50]]
```

**Use case:** Build neural-network layers and numerical kernels with built-in matrix math instead of importing NumPy.

### Automatic Differentiation

The `ad` module computes exact gradients via forward-mode dual numbers, including chain-rule-aware transcendental functions.

```
let grad_x2 = ad.grad(|x| x * x, 3.0);          // 6.0
let grad_sin = ad.grad(|x| ad.sin(x), 0.0);     // 1.0
let grad_exp_x2 = ad.grad(|x| ad.exp(x * x), 1.0);
```

**Use case:** Train models or run optimization and physics simulations needing exact gradients without hand-deriving formulas.

### Lazy ML Pipelines

The `pipeline` module builds chainable, lazily-evaluated data flows for filtering, mapping, batching, shuffling, and sampling datasets.

```
let batches = pipeline.from(raw_samples)
    .map(|s| tensor.from(s))
    .map(|t| t / 255.0)        // normalize
    .shuffle()
    .batch(64)
    .collect();
```

**Use case:** Preprocess and batch training data for ML models in a single readable, deferred pipeline.

### Model I/O & Provenance

Save and load tensor weight maps in the native WKTM format, attach provenance metadata, and import external `GGUF` (llama.cpp) and `ONNX` weights.

```
model.save(weights, "my_model.wktm");
model.setMeta(weights, "author", "WakaDemo");
let loaded = model.load("my_model.wktm");
let llama = model.loadGGUF("llama-7b.gguf");
let onnx = model.loadONNX("resnet50.onnx");
```

**Use case:** Persist trained checkpoints with metadata and load pretrained LLM or vision weights from standard formats.

### GPU Dispatch (OpenCL)

The `gpu` module detects OpenCL devices and runs compute kernels on NVIDIA, AMD, Intel, or Apple GPUs, falling back gracefully when unavailable.

```
if gpu.available() {
    let devices = gpu.devices();
    for d in devices { println(f"GPU: {d}"); }
    let result = gpu.run(kernel);
}
```

**Use case:** Offload heavy vectorized compute to the GPU while keeping a CPU fallback for portability.

### Built-in Statistics & Scikit-style ML

The `stats` module offers dependency-free descriptive statistics, while `ml` wraps regression, clustering, and train/test splitting.

```
println(f"Mean: {stats.mean(data):.3f}");
println(f"Correlation: {stats.correlation(xs, ys):.4f}");
let split = ml.trainTestSplit(X_data, y_data, test_size: 0.2);
let lr = ml.LinearRegression();
lr.fit(split[0], split[2]);
let km = ml.KMeans(3);
km.fit(cluster_data);
```

**Use case:** Run quick exploratory analysis and fit regression or clustering models directly in WakaWaka.

## Data & SQL Integration

### Injection-Safe SQL Literals

The `@sql`...`` literal connects via `sql.connect` and treats every `{variable}` as a bound parameter — string concatenation never happens, so SQL injection is impossible.

```
let db = sql.connect("sqlite:demo.db");
let min_age = 28;
let filtered = db.query(@sql`
    SELECT name, age, score FROM users
    WHERE age >= {min_age}
    ORDER BY score DESC
`);
for u in filtered { println(f"  {u.name}: {u.score}"); }
```

**Use case:** Query user-supplied filters against a database with zero risk of injection attacks.

### Atomic Transactions

`db.transaction` takes a closure that auto-commits on success and auto-rolls-back on error, keeping multi-statement updates consistent.

```
db.transaction(|tx| {
    tx.exec(@sql`UPDATE users SET score = score - {points} WHERE id = {from_id}`);
    tx.exec(@sql`UPDATE users SET score = score + {points} WHERE id = {to_id}`);
});
```

**Use case:** Perform safe transfers or balance updates that must all succeed or all fail together.

## Shell & Systems

### Shell Command Literals

The `$`...`` literal runs a shell command and captures stdout as a string, with chainable methods like `.trim()` and `.lines()`.

```
let cwd = $`pwd`.trim();
let listing = $`ls -la`.lines();
let wk_files = $`find examples -name "*.wk"`.trim().lines();
println(f"Found {wk_files.len()} .wk files");
```

**Use case:** Write glue and automation scripts that drive system tools and process their output natively.

### Safe Variable Interpolation

Variables interpolated into shell literals are auto-escaped, preventing command injection even with hostile input.

```
let filename = "my file.txt";
let content = $`cat {filename}`;       // spaces handled safely

let user_input = "'; rm -rf /; echo '";
$`echo {user_input}`;                  // no injection
```

**Use case:** Safely pass untrusted filenames or arguments to shell commands in build and ops tooling.

## Interop & Foreign Modules

### Python Imports & pyblock

Import any Python module with `import python` and call it directly, or embed raw Python with `pyblock` — variables flow both ways.

```
import python "numpy" as np;
let arr = np.array([1, 2, 3, 4, 5]);
println(f"Mean: {np.mean(arr)}");

let numbers = [1, 2, 3, 4, 5];
pyblock {
    import statistics
    result = f"Mean = {statistics.mean(numbers)}"
}
println(result);
```

**Use case:** Reuse the entire Python ecosystem (NumPy, pandas, scikit-learn) without leaving WakaWaka.

### Runtime FFI: Python, JVM & Node.js

The `py`, `jvm`, and `node` modules dynamically load runtimes via `dlopen` at runtime, each reporting `.available()` and gracefully degrading when absent.

```
if jvm.available() {
    let System = jvm.import("java.lang.System");
    let time = jvm.call(System, "currentTimeMillis");
    println(f"Java time: {time}");
}
```

**Use case:** Call into existing Java or Node.js libraries from compiled WakaWaka with zero build-time dependencies.

### Native Wakawaka Modules

Split code across files and import other `.wk` modules under a namespace; the compiler builds per-module namespace maps to avoid collisions.

```
import "mymath.wk" as mm;

fn main() {
    println(f"add(3, 4) = {mm.add(3, 4)}");
    println(f"PI = {mm.PI}");
}
main();
```

**Use case:** Organize a larger project into reusable library modules that work in both interpreted and compiled mode.

## Metaprogramming & Compile-Time

### Compile-Time Evaluation (comptime)

`comptime { ... }` blocks are evaluated at compile time, folding constant expressions into the binary for zero runtime cost.

```
const TWO_PI = comptime { 2 * 3.14159265358979 };
const BIG = comptime { 100 * 100 + 42 };
println(f"TWO_PI: {TWO_PI}");
```

**Use case:** Precompute lookup tables or derived constants once at build time rather than on every run.

### Macros

The `macro` keyword defines reusable code-expanding templates that are substituted at their call sites.

```
macro square(x) { x * x }
macro cube(x)   { x * x * x }

println(f"square(5) = {square(5)}");
println(f"cube(3) = {cube(3)}");
```

**Use case:** Eliminate boilerplate for small repeated expressions without function-call overhead.

### Gradual Type Checking

Types are inferred everywhere but can be annotated explicitly; `waka check` statically analyzes a program without running it.

```
let x: int = 42;
let y: float = 3.14;
let msg: str = "hello";
// Run: waka check examples/ai_demo.wk
```

**Use case:** Catch type errors in CI or pre-commit hooks while keeping rapid, annotation-light development.

## Built-in Web Server

### Dynamic .wk Pages

`waka serve` serves static files and executes `.wk` files as dynamic pages using `<?= expr ?>` output tags and `<?wk ... ?>` code blocks, with a familiar `request` object.

```
<h1>Hello, <?= request.GET.get("name") ?? "World" ?>!</h1>
<?wk let items = ["Apple", "Banana", "Cherry"]; ?>
<ul>
<?wk for item in items { ?>
  <li><?= item ?></li>
<?wk } ?>
</ul>
```

**Use case:** Stand up small dynamic web apps or admin pages with no external web framework, using PHP-style templating.

## Building for macOS, Linux & Windows

WakaWaka is written in Python and packaged into a single self-contained `waka` executable with PyInstaller. Build natively on each OS — PyInstaller does not cross-compile. The optional `waka build` command (which transpiles a `.wk` program to C99 and compiles it) additionally needs a C compiler — gcc, clang, or MSVC — on that machine.

### Linux

```
# Requires Python 3.10+ and pip
git clone https://github.com/Vern-AllWorks-LLC/wakawaka.git
cd wakawaka
python3 build_waka.py        # bundles with PyInstaller -> dist/waka
./dist/waka run examples/hello.wk
```

**Tip:** Build on an older distro (e.g. Ubuntu 20.04) so the binary runs across a wide range of glibc versions.

### macOS

```
# Requires Python 3.10+  (brew install python)
git clone https://github.com/Vern-AllWorks-LLC/wakawaka.git
cd wakawaka
python3 build_waka.py        # -> dist/waka
```

**Tip:** Build once on Apple Silicon and once on Intel (or with a universal2 Python) for both arches, then codesign + notarize to avoid Gatekeeper warnings.

### Windows

```
:: Requires Python 3.10+ from python.org
git clone https://github.com/Vern-AllWorks-LLC/wakawaka.git
cd wakawaka
python build_waka.py         :: -> dist\waka.exe
```

**Tip:** Sign `waka.exe` with an Authenticode certificate so SmartScreen does not flag an unknown publisher.

### Automated builds (GitHub Actions)

A single workflow with a runner matrix builds every binary on each push and uploads them as release artifacts — no local build machines required.

```
strategy:
  matrix:
    os: [ubuntu-latest, macos-14, macos-13, windows-latest]
runs-on: ${{ matrix.os }}
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-python@v5
    with: { python-version: '3.12' }
  - run: python build_waka.py
  - uses: actions/upload-artifact@v4
    with: { name: waka-${{ matrix.os }}, path: dist/* }
```

**Use case:** Produce macOS, Linux and Windows binaries reproducibly from one push; add signing/notarization steps with your certificates stored as encrypted secrets.

---

## Documentation

Full documentation: <https://vernallworks.com/docs-wakawaka.php>

## License

See the `LICENSE` file.
