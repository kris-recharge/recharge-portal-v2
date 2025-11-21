"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { supabase } from "@/lib/supabaseClient";

type Step = "request" | "verify";

export default function LoginPage() {
  const router = useRouter();

  const [step, setStep] = useState<Step>("request");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");

  const [loading, setLoading] = useState(false);
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Step 1: send OTP to email
  const handleRequestCode = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setInfo(null);

    if (!email) {
      setError("Please enter an email address.");
      return;
    }

    setLoading(true);
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: {
        shouldCreateUser: false, // don’t auto sign up new users
      },
    });
    setLoading(false);

    if (error) {
      console.error("Supabase OTP send error:", error);
      setError(error.message || "Unable to send code. Please try again.");
      return;
    }

    setInfo("Check your email for a 6-digit code.");
    setStep("verify");
  };

  // Step 2: verify OTP and create session
  const handleVerifyCode = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setInfo(null);

    if (!code || code.length !== 6) {
      setError("Please enter the 6-digit code from your email.");
      return;
    }

    setLoading(true);

    const { data, error } = await supabase.auth.verifyOtp({
      email,
      token: code,
      type: "email",
    });
    setLoading(false);

    if (error) {
      console.error("Supabase OTP verify error:", error);
      setError(error.message || "Invalid code. Please try again.");
      return;
    }

    console.log("Supabase OTP login success:", data);
    router.push("/app");
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-slate-950">
      <div className="w-full max-w-md rounded-xl bg-slate-900 shadow-xl border border-slate-800 p-6 space-y-6">
        {/* Logo + heading */}
        <div className="flex flex-col items-center space-y-2">
          <div className="relative h-16 w-28">
            <Image
              src="/ReCharge-Logo_REVA.png"
              alt="ReCharge Alaska logo"
              fill
              className="object-contain"
            />
          </div>
          <h1 className="text-xl font-semibold text-center text-slate-50">
            ReCharge Portal Login
          </h1>
          <p className="text-sm text-slate-400 text-center">
            Enter your email and we&apos;ll send you a 6-digit code to sign in.
          </p>
        </div>

        {step === "request" && (
          <form className="space-y-4" onSubmit={handleRequestCode}>
            <div className="space-y-1">
              <label className="block text-sm font-medium text-slate-200">
                Email
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-50 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                placeholder="you@example.com"
              />
            </div>

            {error && (
              <p className="text-sm text-red-400 text-center">{error}</p>
            )}
            {info && (
              <p className="text-sm text-emerald-400 text-center">{info}</p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-md bg-emerald-600 text-white py-2 text-sm font-medium hover:bg-emerald-700 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? "Sending code…" : "Send 6-digit code"}
            </button>
          </form>
        )}

        {step === "verify" && (
          <form className="space-y-4" onSubmit={handleVerifyCode}>
            <div className="space-y-1">
              <label className="block text-sm font-medium text-slate-200">
                Email
              </label>
              <input
                type="email"
                value={email}
                disabled
                className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-400"
              />
            </div>

            <div className="space-y-1">
              <label className="block text-sm font-medium text-slate-200">
                6-digit code
              </label>
              <input
                type="text"
                inputMode="numeric"
                maxLength={6}
                required
                value={code}
                onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))}
                className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm tracking-[0.4em] text-center font-mono text-slate-50 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                placeholder="123456"
              />
            </div>

            {error && (
              <p className="text-sm text-red-400 text-center">{error}</p>
            )}
            {info && (
              <p className="text-sm text-emerald-400 text-center">{info}</p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-md bg-emerald-600 text-white py-2 text-sm font-medium hover:bg-emerald-700 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? "Verifying…" : "Authenticate"}
            </button>

            <button
              type="button"
              onClick={() => {
                setStep("request");
                setCode("");
                setInfo(null);
                setError(null);
              }}
              className="w-full text-xs text-slate-400 mt-1 hover:text-slate-200"
            >
              Use a different email
            </button>
          </form>
        )}
      </div>
    </main>
  );
}