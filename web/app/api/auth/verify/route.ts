import { NextResponse } from "next/server";
import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

function toAllowedEvseHeaderValue(value: unknown): string {
  // portal_users.allowed_evse_ids is expected to be text[] in Postgres, which comes back as string[]
  // Normalize aggressively and emit a comma-separated list.
  if (!value) return "";
  if (Array.isArray(value)) {
    return value
      .map((v) => String(v).trim())
      .filter((v) => v.length > 0)
      .join(",");
  }
  // If it somehow arrives as a string (e.g., '{a,b}' or 'a,b'), try to normalize.
  const s = String(value).trim();
  if (!s) return "";
  // Strip surrounding braces from Postgres array literal.
  const stripped = s.startsWith("{") && s.endsWith("}") ? s.slice(1, -1) : s;
  return stripped
    .split(",")
    .map((v) => v.replace(/^\"|\"$/g, "").trim())
    .filter((v) => v.length > 0)
    .join(",");
}

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

  // Fetch authorization scope (allowed EVSE IDs) from portal_users.
  // We prefer lookup by email (matches your admin workflow), with a fallback to user_id.
  let allowedEvseHeader = "";
  const userEmail = (data.user.email ?? "").trim().toLowerCase();

  try {
    if (userEmail) {
      const { data: puByEmail, error: puEmailErr } = await supabase
        .from("portal_users")
        .select("allowed_evse_ids")
        .ilike("email", userEmail)
        .maybeSingle();

      if (!puEmailErr && puByEmail) {
        allowedEvseHeader = toAllowedEvseHeaderValue((puByEmail as any).allowed_evse_ids);
      }
    }

    if (!allowedEvseHeader) {
      const { data: puById, error: puIdErr } = await supabase
        .from("portal_users")
        .select("allowed_evse_ids")
        .eq("id", data.user.id)
        .maybeSingle();

      if (!puIdErr && puById) {
        allowedEvseHeader = toAllowedEvseHeaderValue((puById as any).allowed_evse_ids);
      }
    }
  } catch {
    // If anything goes sideways, fail closed on EVSE scope by sending empty header.
    allowedEvseHeader = "";
  }

  res.headers.set("X-Portal-Allowed-EVSE", allowedEvseHeader);

  return res;
}
