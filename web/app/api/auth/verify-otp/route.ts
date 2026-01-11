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

  // Next 16: cookies() is async
  const cookieStore = await cookies();

  // IMPORTANT: set cookies on the response object
  const res = NextResponse.json({ ok: true }, { status: 200 });

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return (cookieStore as any).getAll();
        },
        setAll(cookiesToSet) {
          for (const { name, value, options } of cookiesToSet) {
            // write cookies onto response
            res.cookies.set(name, value, options);
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
    return NextResponse.json({ ok: false, error: error.message }, { status: 401 });
  }

  return res;
}
