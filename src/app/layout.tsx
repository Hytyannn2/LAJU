import type { Metadata } from 'next';
import './globals.css';
import 'leaflet/dist/leaflet.css';

export const metadata: Metadata = {
  title: 'LAJU — AI Transit Network Designer for Kelantan',
  description: 'AI-driven generative public transit network designer optimizing bus coverage, BPR congestion, and DRT equity across Kelantan.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-[#090d16] text-gray-100 overflow-hidden min-h-screen">
        {children}
      </body>
    </html>
  );
}
