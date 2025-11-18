// app/page.tsx
import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-4">
      <h1 className="text-2xl font-semibold">ReCharge Portal v2</h1>
      <p className="text-gray-600">
        This is the new web portal shell. Choose where to go:
      </p>
      <div className="flex gap-3">
        <Link
          href="/login"
          className="px-4 py-2 rounded-md border border-gray-300 hover:bg-gray-50"
        >
          Login
        </Link>
        <Link
          href="/app"
          className="px-4 py-2 rounded-md border border-gray-300 hover:bg-gray-50"
        >
          Go to app
        </Link>
      </div>
    </main>
  );
}