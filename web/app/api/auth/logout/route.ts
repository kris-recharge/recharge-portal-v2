import { NextResponse } from "next/server";
import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

export async function GET(req: Request) {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

  if (!supabaseUrl || !supabaseAnonKey) {
    return new NextResponse("Server misconfigured", { status: 500 });
  }

  // Redirect back to login after signout.
  // IMPORTANT: don't build the redirect URL from `req.url` because behind a reverse proxy
  // it may look like `http://0.0.0.0:3000/...` and the browser will try to follow that.
  // Instead, trust forwarded headers (set by Caddy) and fall back to Host.
  const xfProto = req.headers.get("x-forwarded-proto") || "https";
  const xfHost = req.headers.get("x-forwarded-host") || req.headers.get("host") || "";
  const base = xfHost ? `${xfProto}://${xfHost}` : "";

  const res = NextResponse.redirect(base ? new URL("/login", base) : "/login", { status: 302 });

  const cookieStore = await cookies();

  const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll(cookiesToSet) {
        cookiesToSet.forEach(({ name, value, options }) => {
          res.cookies.set(name, value, options);
        });
      },
    },
  });

  // This clears the Supabase auth cookies via setAll(...)
  await supabase.auth.signOut();

  return res;
}