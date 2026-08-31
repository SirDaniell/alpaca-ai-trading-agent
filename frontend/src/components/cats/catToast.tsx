import { toast } from "sonner";
import { Cat } from "./Cat";
import { useCatBehavior } from "./useCatBehavior";
import { CATS, type CatVariant } from "./catSprites";

function ToastCat({ variant, message }: { variant: CatVariant; message: string }) {
  const { behavior, interrupt } = useCatBehavior(variant, true);

  return (
    <div className="relative flex w-full items-end gap-3 rounded-lg border border-border bg-card px-4 pb-3 pt-6 text-card-foreground shadow-lg">
      <div className="absolute -top-9 left-3">
        <Cat
          variant={variant}
          behavior={behavior}
          size={56}
          onClick={() => interrupt("fight", 900)}
        />
      </div>
      <div className="min-w-0 flex-1 pl-14">
        <p className="text-sm font-medium leading-snug">{message}</p>
        <p className="text-xs text-muted-foreground">{CATS[variant].name} approves</p>
      </div>
    </div>
  );
}

/** Fire a toast with a cat perched on top of it. */
export function catToast(message: string, variant: CatVariant = "marmalade", duration = 6000) {
  return toast.custom(() => <ToastCat variant={variant} message={message} />, { duration });
}
