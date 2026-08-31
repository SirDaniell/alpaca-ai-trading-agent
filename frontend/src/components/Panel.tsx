import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Panel({
  title,
  subtitle,
  actions,
  className,
  bodyClassName,
  children,
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
}) {
  return (
    <section className={cn("glass rounded-3xl", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 px-4 py-3 sm:px-5">
          <div>
            {title && (
              <h2 className="text-[13px] font-semibold tracking-[0.14em] text-muted-foreground uppercase">
                {title}
              </h2>
            )}
            {subtitle && <p className="mt-0.5 text-xs text-muted-foreground/80">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className={cn("p-4 sm:p-5", bodyClassName)}>{children}</div>
    </section>
  );
}

export function Metric({
  label,
  value,
  sub,
  tone = "neutral",
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "neutral" | "up" | "down";
  accent?: "blue" | "gold" | "silver" | undefined;
}) {
  return (
    <div className="glass rounded-2xl px-4 py-3.5">
      <div className="flex items-center gap-2">
        {accent && (
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              accent === "blue" && "bg-primary",
              accent === "gold" && "bg-gold",
              accent === "silver" && "bg-silver",
            )}
          />
        )}
        <p className="text-[11px] font-medium tracking-[0.12em] text-muted-foreground uppercase">
          {label}
        </p>
      </div>
      <p
        className={cn(
          "num mt-2 text-xl font-semibold sm:text-2xl",
          tone === "up" && "text-bull",
          tone === "down" && "text-bear",
        )}
      >
        {value}
      </p>
      {sub && (
        <p
          className={cn(
            "num mt-0.5 text-xs",
            tone === "up" ? "text-bull/80" : tone === "down" ? "text-bear/80" : "text-muted-foreground",
          )}
        >
          {sub}
        </p>
      )}
    </div>
  );
}
