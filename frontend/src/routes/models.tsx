import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/AppShell";
import { ModelsPanel } from "@/components/models/ModelsPanel";

export const Route = createFileRoute("/models")({
  head: () => ({
    meta: [
      { title: "AXE Genesis — Model Registry & Trainer" },
      {
        name: "description",
        content:
          "Manage AXE Genesis reinforcement-learning checkpoints: pretrain on synthetic or real data, hot-swap the active model, view evaluations, delete checkpoints.",
      },
      { property: "og:title", content: "AXE Genesis — Model Registry & Trainer" },
      {
        property: "og:description",
        content: "Pretrain, activate and evaluate RL model checkpoints for the AXE Genesis options agent.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ModelsPage,
});

function ModelsPage() {
  return (
    <AppShell>
      <ModelsPanel />
    </AppShell>
  );
}
