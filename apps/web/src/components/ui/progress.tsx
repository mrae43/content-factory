import * as React from "react"

import { cn } from "@/lib/utils"

function Progress({
  value,
  max = 100,
  className,
  ...props
}: React.ComponentProps<"div"> & { value?: number; max?: number }) {
  const percentage =
    value != null ? Math.min(100, Math.max(0, (value / max) * 100)) : 0

  return (
    <div
      data-slot="progress-root"
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={value}
      className={cn(
        "relative h-2 w-full overflow-hidden rounded-full bg-primary/20",
        className
      )}
      {...props}
    >
      <div
        data-slot="progress-indicator"
        className="h-full rounded-full bg-primary transition-all duration-300 ease-in-out"
        style={{ width: `${percentage}%` }}
      />
    </div>
  )
}

export { Progress }
