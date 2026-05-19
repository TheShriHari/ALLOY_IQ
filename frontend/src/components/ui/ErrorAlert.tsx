export function ErrorAlert({ message }: { message: string }) {
  return (
    <div className="p-4 rounded-xl border border-red-500/20 bg-red-500/10">
      <p className="text-red-400 text-sm">⚠ {message}</p>
    </div>
  );
}
