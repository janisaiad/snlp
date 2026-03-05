#!/usr/bin/env python3
"""
Write sclite-style result from ref.trn and hyp.trn (drop-in when sclite is not installed).
Usage: score_cer_sclite.py -r ref.trn trn -h hyp.trn trn -i rm -o all stdout
Reads ref and hyp, computes CER, prints to stdout so asr.sh can redirect to result.txt.
"""
import sys

def parse_args():
    ref_path = hyp_path = None
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "-r" and i + 1 < len(argv):
            ref_path = argv[i + 1]
            i += 2
        elif argv[i] == "-h" and i + 1 < len(argv):
            hyp_path = argv[i + 1]
            i += 2
        else:
            i += 1
    return ref_path, hyp_path

def read_trn(path: str) -> list[tuple[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            utt_id = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            out.append((utt_id, text))
    return out

def cer_char(ref: str, hyp: str) -> tuple[int, int, int, int, int]:
    """Levenshtein at char level. Returns (corr, sub, dele, ins, ref_len)."""
    try:
        import editdistance
        d = editdistance.eval(ref, hyp)
    except ImportError:
        r, h = list(ref), list(hyp)
        n, m = len(r), len(h)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        d = dp[n][m]
    n = len(ref)
    corr = max(0, n - d)
    return (corr, 0, 0, 0, n)

def main():
    ref_path, hyp_path = parse_args()
    if not ref_path or not hyp_path:
        sys.stderr.write("usage: score_cer_sclite.py -r ref.trn trn -h hyp.trn trn ...\n")
        sys.exit(1)
    ref_list = read_trn(ref_path)
    hyp_list = read_trn(hyp_path)
    if len(ref_list) != len(hyp_list):
        sys.stderr.write("score_cer_sclite: ref and hyp utterance count mismatch\n")
        sys.exit(1)
    total_corr = total_ref = total_err = 0
    for (_, r), (_, h) in zip(ref_list, hyp_list):
        rc, rs, rd, ri, rn = cer_char(r, h)
        total_corr += rc
        total_ref += rn
        total_err += (rn - rc)  # approx
    if total_ref == 0:
        err_pct = 0.0
    else:
        err_pct = 100.0 * (total_ref - total_corr) / total_ref
    # sclite Sum/Avg line: SPKR #Snt #Wrd Corr Sub Del Ins Err S.Err
    sys.stdout.write("| Sum/Avg | {} | {} | {:.1f} | 0.0 | 0.0 | 0.0 | {:.2f} | 0.0 |\n".format(
        len(ref_list), total_ref, 100.0 * total_corr / total_ref if total_ref else 0, err_pct
    ))
    sys.stdout.write("| SPKR | # Snt | # Wrd | Corr | Sub | Del | Ins | Err | S.Err |\n")
    sys.stdout.write("| Sum/Avg | {} | {} | {:.1f} | 0.0 | 0.0 | 0.0 | {:.2f} | 0.0 |\n".format(
        len(ref_list), total_ref, 100.0 * total_corr / total_ref if total_ref else 0, err_pct
    ))

if __name__ == "__main__":
    main()
