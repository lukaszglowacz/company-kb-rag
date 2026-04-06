import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Company KB RAG",
  description: "Company knowledge base RAG chatbot",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="h-screen bg-white text-gray-900 antialiased">
        {children}
      </body>
    </html>
  );
}
