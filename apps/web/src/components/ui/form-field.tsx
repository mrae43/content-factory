import * as React from "react"

import { cn } from "@/lib/utils"
import { Label } from "@/components/ui/label"

interface FormFieldProps extends React.ComponentProps<"div"> {
  label: string
  htmlFor?: string
  error?: string
  hint?: string
  required?: boolean
  children: React.ReactNode
}

function FormField({
  label,
  htmlFor,
  error,
  hint,
  required,
  children,
  className,
  ...props
}: FormFieldProps) {
  const hasError = !!error

  return (
    <div
      data-slot="form-field"
      className={cn("space-y-2", className)}
      {...props}
    >
      <Label htmlFor={htmlFor}>
        {label}
        {required && <span className="ml-0.5 text-destructive">*</span>}
      </Label>
      {React.Children.map(children, (child) => {
        if (!React.isValidElement(child)) return child
        return React.cloneElement(child as React.ReactElement<Record<string, unknown>>, {
          "aria-invalid": hasError || undefined,
        })
      })}
      {hasError && (
        <p data-slot="form-field-error" className="text-xs text-destructive">
          {error}
        </p>
      )}
      {hint && !hasError && (
        <p data-slot="form-field-hint" className="text-xs text-muted-foreground">
          {hint}
        </p>
      )}
    </div>
  )
}

export { FormField, type FormFieldProps }
