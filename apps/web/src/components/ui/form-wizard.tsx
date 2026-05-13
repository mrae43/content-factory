"use client"

import * as React from "react"
import { CheckIcon } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"

interface WizardStep {
  title: string
  description?: string
}

interface FormWizardProps extends React.ComponentProps<"div"> {
  steps: WizardStep[]
  currentStep: number
  onNext?: () => void
  onBack?: () => void
  onComplete?: () => void
  isPending?: boolean
  nextLabel?: string
  backLabel?: string
  completeLabel?: string
  canNext?: boolean
  children: React.ReactNode
}

function WizardStepIndicator({
  steps,
  currentStep,
}: {
  steps: WizardStep[]
  currentStep: number
}) {
  return (
    <nav aria-label="Form progress" data-slot="wizard-steps">
      <ol className="flex items-center gap-1">
        {steps.map((step, index) => {
          const isCompleted = index < currentStep
          const isCurrent = index === currentStep

          return (
            <React.Fragment key={index}>
              {index > 0 && (
                <li
                  aria-hidden="true"
                  className={cn(
                    "h-px flex-1 min-w-4",
                    index <= currentStep ? "bg-primary" : "bg-border"
                  )}
                />
              )}
              <li className="flex items-center gap-2">
                <div
                  className={cn(
                    "size-7 rounded-full flex items-center justify-center text-xs font-medium border-2 transition-colors shrink-0",
                    isCompleted &&
                      "bg-primary text-primary-foreground border-primary",
                    isCurrent &&
                      "border-primary text-primary bg-transparent",
                    !isCompleted &&
                      !isCurrent &&
                      "border-border text-muted-foreground bg-transparent"
                  )}
                  aria-current={isCurrent ? "step" : undefined}
                >
                  {isCompleted ? (
                    <CheckIcon className="size-3.5" />
                  ) : (
                    index + 1
                  )}
                </div>
                <span
                  className={cn(
                    "text-sm hidden sm:inline",
                    (isCompleted || isCurrent)
                      ? "text-foreground font-medium"
                      : "text-muted-foreground"
                  )}
                >
                  {step.title}
                </span>
              </li>
            </React.Fragment>
          )
        })}
      </ol>
    </nav>
  )
}

function FormWizard({
  steps,
  currentStep,
  onNext,
  onBack,
  onComplete,
  isPending = false,
  nextLabel = "Next",
  backLabel = "Back",
  completeLabel = "Submit",
  canNext = true,
  children,
  className,
  ...props
}: FormWizardProps) {
  const isFirst = currentStep === 0
  const isLast = currentStep === steps.length - 1

  return (
    <div data-slot="form-wizard" className={cn("space-y-6", className)} {...props}>
      <WizardStepIndicator steps={steps} currentStep={currentStep} />
      <div data-slot="wizard-content">{children}</div>
      <div data-slot="wizard-actions" className="flex justify-between pt-2">
        <Button
          variant="outline"
          onClick={onBack}
          disabled={isFirst || isPending}
          type="button"
        >
          {backLabel}
        </Button>
        {isLast ? (
          <Button
            onClick={onComplete}
            disabled={isPending}
            type="button"
          >
            {isPending && <Spinner className="mr-1.5" />}
            {completeLabel}
          </Button>
        ) : (
          <Button
            onClick={onNext}
            disabled={!canNext || isPending}
            type="button"
          >
            {nextLabel}
          </Button>
        )}
      </div>
    </div>
  )
}

export {
  FormWizard,
  WizardStepIndicator,
  type WizardStep,
  type FormWizardProps,
}
