// app/page.tsx
import { redirect } from "next/navigation";

export default function Home() {
  // Immediately send users to the login page when they hit the root URL
  redirect("/login");

  // This return is never actually rendered because of the redirect,
  // but it satisfies the component contract.
  return null;
}