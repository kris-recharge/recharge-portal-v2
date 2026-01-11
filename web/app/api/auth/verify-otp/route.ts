import { NextResponse } from "next/server";
import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

export async function POST(req: Request) {
  const { email, token } = await req.json();

  if (!email || !token) {
    return NextResponse.json(
      { ok: false, error: "missing email/token" },
      { status: 400 }
    );
  }

  // Next 15/16: cookies() is async
  const cookieStore = await cookies();

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          // cookieStore is ReadonlyRequestCookies; in practice getAll exists at runtime.
          // Type differences across Next versions can be annoying, so we cast.
          return (cookieStore as any).getAll();
        },
        setAll(cookiesToSet) {
          for (const { name, value, options } of cookiesToSet) {
            (cookieStore as any).set(name, value, options);
          }
        },
      },
    }
  );

  const { error } = await supabase.auth.verifyOtp({
    email,
    token,
    type: "email",
  });

  if (error) {
    return NextResponse.json(
      { ok: false, error: error.message },
      { status: 401 }
    );
  }

  return NextResponse.json({ ok: true }, { status: 200 });
}
