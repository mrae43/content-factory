import * as React from "react"

import { cn } from "@/lib/utils"

function Timeline({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="timeline"
      className={cn("relative", className)}
      {...props}
    />
  )
}

const statusDotStyles: Record<string, string> = {
  completed: "bg-success border-success",
  active: "bg-primary border-primary ring-2 ring-ring/20",
  pending: "bg-muted-foreground/30 border-muted-foreground/30",
  error: "bg-destructive border-destructive",
}

interface TimelineItemProps extends React.ComponentProps<"div"> {
  title: string
  description?: string
  time?: string
  status?: "completed" | "active" | "pending" | "error"
  icon?: React.ReactNode
}

function TimelineItem({
  title,
  description,
  time,
  status = "pending",
  icon,
  className,
  ...props
}: TimelineItemProps) {
  return (
    <div
      data-slot="timeline-item"
      className={cn("relative flex gap-3 pb-6 last:pb-0", className)}
      {...props}
    >
      <div className="relative flex flex-col items-center">
        <div
          className={cn(
            "size-3 rounded-full border-2 shrink-0 mt-1 transition-colors",
            icon ? "bg-transparent" : statusDotStyles[status]
          )}
        >
          {icon && (
            <span className="absolute -translate-x-1/2 -translate-y-1/2 top-1/2 left-1/2">
              {icon}
            </span>
          )}
        </div>
        <div className="w-px flex-1 bg-border" />
      </div>
      <div className="flex-1 min-w-0 -mt-0.5">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium">{title}</p>
          {time && (
            <span className="text-xs text-muted-foreground">{time}</span>
          )}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
        )}
      </div>
    </div>
  )
}

export { Timeline, TimelineItem, type TimelineItemProps }
