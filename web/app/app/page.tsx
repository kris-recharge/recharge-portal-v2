import { redirect } from "next/navigation";
import { cookies } from "next/headers";
import { createServerClient } from "@supabase/ssr";

export default async function AppPage() {
  // Next 16 typing: treat cookies() as async + cast getAll/set for stability
  const cookieStore = await cookies();

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
            (cookieStore as any).set(name, value, options);
          }
        },
      },
    }
  );

  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  return (
    <main style={{ padding: 24 }}>
      <h1>ReCharge Alaska Dashboard</h1>
      <p>Logged in as {user.email}</p>
    </main>
  );
}
