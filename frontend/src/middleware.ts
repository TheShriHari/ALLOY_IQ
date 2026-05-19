import { NextRequest, NextResponse } from "next/server";

const PROTECTED = ["/predict", "/inverse", "/microstructure", "/history", "/compare"];

export function middleware(request: NextRequest) {
  const token = request.cookies.get("alloyiq_token")?.value;
  const isProtected = PROTECTED.some(p => request.nextUrl.pathname.startsWith(p));
  
  if (isProtected && !token) {
    return NextResponse.redirect(new URL("/auth/login", request.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/predict/:path*", "/inverse/:path*", "/microstructure/:path*", "/history/:path*", "/compare/:path*"],
};
