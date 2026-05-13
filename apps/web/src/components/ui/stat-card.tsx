import * as React from "react"
import { TrendingUpIcon, TrendingDownIcon, MinusIcon } from "lucide-react"

import { cn } from "@/lib/utils"
import { Card, CardContent } from "@/components/ui/card"

interface StatCardTrend {
  value: number
  direction: "up" | "down" | "neutral"
  label?: string
}

interface StatCardProps extends React.ComponentProps<typeof Card> {
  label: string
  value: React.ReactNode
  icon?: React.ReactNode
  trend?: StatCardTrend
}

const trendIcons = {
  up: TrendingUpIcon,
  down: TrendingDownIcon,
  neutral: MinusIcon,
} as const

function StatCard({
  label,
  value,
  icon,
  trend,
  className,
  ...props
}: StatCardProps) {
  return (
    <Card
      data-slot="stat-card"
      className={cn("relative overflow-hidden", className)}
      {...props}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium text-muted-foreground">{label}</p>
            <p className="text-2xl font-bold tabular-nums">{value}</p>
          </div>
          {icon && (
            <div className="size-10 rounded-lg bg-muted flex items-center justify-center text-muted-foreground">
              {icon}
            </div>
          )}
        </div>
        {trend && (() => {
          const TrendIcon = trendIcons[trend.direction]
          return (
            <div
              className={cn(
                "mt-2 flex items-center gap-1 text-xs",
                trend.direction === "up" && "text-success",
                trend.direction === "down" && "text-destructive",
                trend.direction === "neutral" && "text-muted-foreground"
              )}
            >
              <TrendIcon className="size-3" />
              <span>{trend.value}%</span>
              {trend.label && (
                <span className="text-muted-foreground">{trend.label}</span>
              )}
            </div>
          )
        })()}
      </CardContent>
    </Card>
  )
}

export { StatCard, type StatCardTrend, type StatCardProps }
