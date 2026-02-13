import { NextResponse } from "next/server";
import { createServerClient } from "@supabase/ssr";
import { createClient } from "@supabase/supabase-js";
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

  // NOTE: your VPS `.env` currently uses `SUPABASE_SERVICE_ROLL_KEY` (ROLL),
  // but the conventional name is `SUPABASE_SERVICE_ROLE_KEY` (ROLE).
  // Accept either so a small typo doesn't silently disable authorization.
  const supabaseServiceRoleKey =
    process.env.SUPABASE_SERVICE_ROLE_KEY ||
    process.env.SUPABASE_SERVICE_ROLL_KEY ||
    process.env.NEXT_PUBLIC_SUPABASE_SERVICE_ROLE_KEY ||
    process.env.NEXT_PUBLIC_SUPABASE_SERVICE_ROLL_KEY ||
    "";

  if (!supabaseUrl || !supabaseAnonKey) {
    return new NextResponse("Server misconfigured", { status: 500 });
  }
  // Service role is optional, but without it `portal_users` lookups may be blocked by RLS.
  // When missing, we fail closed (no EVSE access) but keep authentication working.

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
  res.headers.set("X-Portal-Logout-Url", "/api/auth/logout");

  // Fetch authorization scope (allowed EVSE IDs) from portal_users.
  // We prefer lookup by email (matches your admin workflow).
  // IMPORTANT: portal_users is usually protected by RLS, so we use the service role key on the server.
  let allowedEvseHeader = "";
  let portalUserFound = false;
  const userEmail = (data.user.email ?? "").trim().toLowerCase();

  try {
    if (supabaseServiceRoleKey && userEmail) {
      const admin = createClient(supabaseUrl, supabaseServiceRoleKey, {
        auth: { persistSession: false, autoRefreshToken: false },
      });

      // Primary: exact match on normalized email.
      const { data: puByEmail, error: puEmailErr } = await admin
        .from("portal_users")
        .select("email, allowed_evse_ids")
        .eq("email", userEmail)
        .maybeSingle();

      if (!puEmailErr && puByEmail) {
        portalUserFound = true;
        allowedEvseHeader = toAllowedEvseHeaderValue((puByEmail as any).allowed_evse_ids);
      }

      // Fallback: case-insensitive match (helps if DB stored mixed-case emails).
      if (!portalUserFound) {
        const { data: puByEmail2, error: puEmailErr2 } = await admin
          .from("portal_users")
          .select("email, allowed_evse_ids")
          .ilike("email", userEmail)
          .maybeSingle();

        if (!puEmailErr2 && puByEmail2) {
          portalUserFound = true;
          allowedEvseHeader = toAllowedEvseHeaderValue((puByEmail2 as any).allowed_evse_ids);
        }
      }
    }
  } catch {
    allowedEvseHeader = "";
    portalUserFound = false;
  }

  // Fail closed: empty allowlist means "no EVSE access" in Streamlit.
  // IMPORTANT: NULL or [] in portal_users.allowed_evse_ids should also result in empty allowlist.
  // We also expose a debug/header flag so it’s obvious whether the portal_users row was found.
  res.headers.set("X-Portal-User-Found", portalUserFound ? "1" : "0");

  // Primary header used by Streamlit/Caddy wiring (comma-separated EVSE IDs)
  res.headers.set("X-Portal-Allowed-Evse-Ids", allowedEvseHeader);

  // Back-compat / alias (some places used this older name)
  res.headers.set("X-Portal-Allowed-EVSE", allowedEvseHeader);

  // Debug mirrors (handy in browser devtools)
  res.headers.set("X-Debug-Portal-Allowed-Evse", allowedEvseHeader);
  res.headers.set("X-Debug-Portal-Email", userEmail);
  res.headers.set("X-Debug-Portal-UserId", data.user.id);

  return res;
}
