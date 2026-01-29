import { NextResponse } from "next/server";
import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

export async function GET(req: Request) {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

  if (!supabaseUrl || !supabaseAnonKey) {
    return new NextResponse("Server misconfigured", { status: 500 });
  }

  // Important: do not allow caches to store auth decisions.
  const res = NextResponse.json({ ok: true }, { status: 200 });
  res.headers.set("Cache-Control", "no-store, max-age=0");

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

  const { data, error } = await supabase.auth.getUser();

  if (error || !data?.user) {
    // Forward-auth callers (Caddy) use the status code; browsers may follow Location.
    return new NextResponse("Unauthorized", {
      status: 401,
      headers: {
        "Cache-Control": "no-store, max-age=0",
        Location: "/login",
      },
    });
  }

  // Make the authenticated user available to upstream services via headers.
  // Caddy can copy these headers from the auth response to the proxied request.
  res.headers.set("X-Portal-User-Id", data.user.id);
  res.headers.set("X-Portal-User-Email", data.user.email ?? "");

  return res;
}
