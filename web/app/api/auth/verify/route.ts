import { NextResponse } from "next/server";
import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

export async function GET(req: Request) {
  const cookieStore = cookies();

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          for (const { name, value, options } of cookiesToSet) {
            cookieStore.set(name, value, options);
          }
        },
      },
    }
  );

  const { data } = await supabase.auth.getUser();

  // If not authenticated, redirect to /login (so forward_auth results in a browser redirect)
  if (!data?.user) {
    const url = new URL("/login", req.url);
    return NextResponse.redirect(url, 302);
  }

  // Authenticated
  return NextResponse.json({ ok: true }, { status: 200 });
}

