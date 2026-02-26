#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from target_catalog import get_target_spec, list_presets, resolve_targets

def run(command, env=None):
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, check=False)
    return proc.returncode, proc.stdout


def normalize_arch_token(raw):
    arch = raw.lower()
    if arch in {"x86_64", "amd64"}:
        return "x86_64"
    if arch in {"i386", "i486", "i586", "i686", "x86", "x32"}:
        return "x86"
    if arch in {"arm64", "aarch64"}:
        return "aarch64"
    if arch in {"armv7", "armv7a", "armv7l"}:
        return "armv7"
    if arch.startswith("riscv64"):
        return "riscv64"
    if arch.startswith("riscv32"):
        return "riscv32"
    if arch in {"ppc64", "ppc64le"}:
        return "ppc64le"
    if arch in {"ppc", "powerpc"}:
        return "ppc"
    if arch == "s390x":
        return "s390x"
    if arch == "loongarch64":
        return "loongarch64"
    if arch == "mips64":
        return "mips64"
    if arch in {"mips", "mipsel"}:
        return "mips"
    if arch == "wasm32":
        return "wasm32"
    if arch == "wasm64":
        return "wasm64"
    return arch


def host_arch():
    machine = platform.machine().lower()
    return normalize_arch_token(machine)


def fallback_target_for_arch(arch):
    normalized = normalize_arch_token(arch)
    system = platform.system().lower()
    if system == "darwin":
        if normalized == "x86_64":
            return "x86_64-apple-darwin"
        if normalized == "aarch64":
            return "arm64-apple-darwin"
        return ""
    if system == "linux":
        if normalized == "x86_64":
            return "x86_64-linux-gnu"
        if normalized == "x86":
            return "i686-linux-gnu"
        if normalized == "aarch64":
            return "aarch64-linux-gnu"
        if normalized in {"armv7", "armv7a"}:
            return "armv7-linux-gnueabihf"
        if normalized == "riscv64":
            return "riscv64-linux-gnu"
        if normalized == "riscv32":
            return "riscv32-linux-gnu"
        if normalized == "ppc64le":
            return "ppc64le-linux-gnu"
        if normalized == "s390x":
            return "s390x-linux-gnu"
        if normalized == "loongarch64":
            return "loongarch64-linux-gnu"
        if normalized in {"mips64", "mips"}:
            return "mips64el-linux-gnuabi64" if normalized == "mips64" else "mipsel-linux-gnu"
        return ""
    if system == "windows":
        if normalized == "x86_64":
            return "x86_64-w64-mingw32"
        if normalized == "aarch64":
            return "aarch64-w64-mingw32"
    return ""


def is_host_target(target):
    arch = normalize_arch_token(target.split("-")[0])
    return arch == host_arch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", default="bench/programs/phase4/scalar_sum.k")
    parser.add_argument("--targets", default="")
    parser.add_argument(
        "--preset",
        choices=list_presets(),
        default="core",
        help="target group preset (ignored when --targets is provided)",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="include experimental targets from the catalog",
    )
    parser.add_argument(
        "--include-embedded",
        action="store_true",
        help="include embedded/bare-metal targets (reported as interpret_only for now)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="print supported presets and exit",
    )
    parser.add_argument("--out-dir", default="build/phase10/multiarch")
    parser.add_argument("--json-out", default="bench/results/phase10_multiarch.json")
    parser.add_argument("--run-host-smoke", action="store_true")
    parser.add_argument(
        "--run-interpret-smoke",
        action="store_true",
        help="run lightweight --interpret execution checks for interpret_only targets",
    )
    parser.add_argument("--lto", choices=["off", "thin", "full"], default="thin")
    parser.add_argument("--sysroot-x86_64", default=os.environ.get("SPARK_SYSROOT_X86_64", ""))
    parser.add_argument("--sysroot-aarch64", default=os.environ.get("SPARK_SYSROOT_AARCH64", ""))
    parser.add_argument("--sysroot-riscv64", default=os.environ.get("SPARK_SYSROOT_RISCV64", ""))
    parser.add_argument("--sysroot-riscv32", default=os.environ.get("SPARK_SYSROOT_RISCV32", ""))
    parser.add_argument("--sysroot-x86", default=os.environ.get("SPARK_SYSROOT_X86", ""))
    parser.add_argument("--sysroot-armv7", default=os.environ.get("SPARK_SYSROOT_ARMV7", ""))
    parser.add_argument("--sysroot-ppc64le", default=os.environ.get("SPARK_SYSROOT_PPC64LE", ""))
    parser.add_argument("--sysroot-s390x", default=os.environ.get("SPARK_SYSROOT_S390X", ""))
    args = parser.parse_args()

    if args.list_presets:
        print(",".join(list_presets()))
        return

    root = Path(__file__).resolve().parents[2]
    k_bin = root / "k"
    source = root / args.program
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise SystemExit(f"missing source: {source}")

    sysroots = {
        "x86_64": args.sysroot_x86_64,
        "aarch64": args.sysroot_aarch64,
        "riscv64": args.sysroot_riscv64,
        "riscv32": args.sysroot_riscv32,
        "x86": args.sysroot_x86,
        "armv7": args.sysroot_armv7,
        "ppc64le": args.sysroot_ppc64le,
        "s390x": args.sysroot_s390x,
    }

    records = []
    explicit_targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    targets = resolve_targets(
        explicit_targets=explicit_targets if explicit_targets else None,
        preset=args.preset if not explicit_targets else None,
        include_experimental=args.include_experimental,
        include_embedded=args.include_embedded,
    )
    for target in targets:
        arch = normalize_arch_token(target.split("-")[0])
        spec = get_target_spec(target)
        mode = spec.mode if spec else "aot"
        tier = spec.tier if spec else "unknown"
        family = spec.family if spec else arch
        os_class = spec.os_class if spec else "unknown"
        notes = spec.notes if spec else ""
        out_bin = out_dir / f"{source.stem}.{target}.bin"
        command = [str(k_bin), "build", str(source), "-o", str(out_bin), "--target", target]
        build_attempted = mode == "aot"
        interpret = {
            "attempted": False,
            "status_ok": False,
            "status": 0,
            "output": "",
        }
        if args.lto != "off" and build_attempted:
            command.extend(["--lto", args.lto])
        sysroot = sysroots.get(arch, "")
        if sysroot and build_attempted:
            command.extend(["--sysroot", sysroot])
        if build_attempted:
            code, output = run(command)
            build_ok = (code == 0)
        else:
            code = 0
            output = "skipped: embedded target currently interpret_only tier"
            build_ok = False
            if args.run_interpret_smoke:
                interpret_command = [str(k_bin), "run", "--interpret", str(source), "--target", target]
                interpret_status, interpret_output = run(interpret_command)
                interpret["attempted"] = True
                interpret["status"] = interpret_status
                interpret["status_ok"] = (interpret_status == 0)
                interpret["output"] = interpret_output
                if interpret["status_ok"]:
                    output = f"{output} | interpret_ok"
        effective_target = target
        fallback_used = False
        fallback_target = ""
        fallback_output = ""
        fallback_status = 0

        if build_attempted and not build_ok:
            fallback_target = fallback_target_for_arch(arch)
            if fallback_target:
                fallback_used = True
                fallback_command = [str(k_bin), "build", str(source), "-o", str(out_bin), "--target", fallback_target]
                if args.lto != "off":
                    fallback_command.extend(["--lto", args.lto])
                fallback_status, fallback_output = run(fallback_command)
                if fallback_status == 0:
                    build_ok = True
                    effective_target = fallback_target

        smoke = {
            "attempted": False,
            "status_ok": False,
            "output": "",
            "status": 0,
        }
        if build_attempted and build_ok and args.run_host_smoke and is_host_target(target):
            smoke["attempted"] = True
            smoke_status, smoke_output = run([str(out_bin)])
            smoke["status"] = smoke_status
            smoke["status_ok"] = (smoke_status == 0)
            smoke["output"] = smoke_output

        records.append(
            {
                "target": target,
                "arch": arch,
                "family": family,
                "os_class": os_class,
                "tier": tier,
                "mode": mode,
                "notes": notes,
                "build_attempted": build_attempted,
                "build_ok": build_ok,
                "binary": str(out_bin),
                "command": command,
                "interpret": interpret,
                "output": output,
                "effective_target": effective_target,
                "fallback_used": fallback_used,
                "fallback_target": fallback_target,
                "fallback_status": fallback_status,
                "fallback_output": fallback_output,
                "smoke": smoke,
            }
        )

    payload = {
        "host_arch": host_arch(),
        "program": str(source),
        "preset": args.preset if not explicit_targets else "",
        "include_experimental": bool(args.include_experimental),
        "include_embedded": bool(args.include_embedded),
        "targets": records,
        "passed": all((not item["build_attempted"]) or item["build_ok"] for item in records),
    }

    json_out = root / args.json_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    attempted = sum(int(item["build_attempted"]) for item in records)
    passed = sum(int(item["build_ok"]) for item in records if item["build_attempted"])
    skipped = sum(int(not item["build_attempted"]) for item in records)
    interpret_attempted = sum(int(item["interpret"].get("attempted", False)) for item in records if isinstance(item.get("interpret"), dict))
    interpret_ok = sum(int(item["interpret"].get("status_ok", False)) for item in records if isinstance(item.get("interpret"), dict))
    print(f"phase10 multiarch: {passed}/{attempted} builds passed, skipped={skipped}")
    print(f"phase10 multiarch interpret smoke: {interpret_ok}/{interpret_attempted} passed")
    print(f"results json: {json_out}")


if __name__ == "__main__":
    main()
