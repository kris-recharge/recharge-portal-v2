import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { createServerClient } from "@supabase/ssr";

/**
 * Auth gate for Caddy forward_auth.
 * Caddy calls:
 *   GET /api/auth/verify
 *
 * Return:
 *   200 if authenticated
 *   401 if not authenticated
 */
export async function GET(_req: Request) {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    return new NextResponse("Server misconfigured", { status: 500 });
  }

  const cookieStore = await cookies();

  const supabase = createServerClient(supabaseUrl, supabaseAnonKey, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll(cookiesToSet) {
        cookiesToSet.forEach(({ name, value, options }) => {
          cookieStore.set(name, value, options);
        });
      },
    },
  });

  const { data, error } = await supabase.auth.getUser();

  if (error || !data?.user) {
    // Caddy will treat non-2xx as "deny".
    // (Optional) a Location header can help when debugging via curl.
    return new NextResponse("Unauthorized", {
      status: 401,
      headers: { Location: "/login" },
    });
  }

  return NextResponse.json({ ok: true }, { status: 200 });
}
