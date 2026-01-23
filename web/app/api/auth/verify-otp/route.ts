import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { createServerClient } from "@supabase/ssr";

/**
 * Verifies a 6-digit email OTP and mints Supabase auth cookies (httpOnly).
 * Your login page calls this endpoint:
 *   POST /api/auth/verify-otp
 *   body: { email: string, token: string }
 */
export async function POST(req: Request) {
  try {
    const { email, token } = await req.json();

    if (!email || typeof email !== "string") {
      return NextResponse.json({ error: "Missing or invalid email." }, { status: 400 });
    }
    if (!token || typeof token !== "string") {
      return NextResponse.json({ error: "Missing or invalid token." }, { status: 400 });
    }

    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseAnonKey) {
      return NextResponse.json(
        { error: "Server misconfigured: missing Supabase env vars." },
        { status: 500 }
      );
    }

    const cookieStore = await cookies();

    const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          // Supabase will provide the cookie names/values/options it needs
          cookiesToSet.forEach(({ name, value, options }) => {
            cookieStore.set(name, value, options);
          });
        },
      },
    });

    const { data, error } = await supabase.auth.verifyOtp({
      email,
      token,
      type: "email", // 6-digit OTP to email
    });

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 401 });
    }

    // Cookies are now set via setAll(). Session should exist after verifyOtp.
    return NextResponse.json(
      {
        ok: true,
        user_id: data.user?.id ?? null,
      },
      { status: 200 }
    );
  } catch (e: any) {
    return NextResponse.json(
      { error: e?.message ?? "Server error verifying OTP." },
      { status: 500 }
    );
  }
}
