declare module "recharts";

declare module "@tanstack/react-router" {
  import type { ComponentType } from "react";

  export const Link: any;
  export const Outlet: ComponentType<any>;
  export const HeadContent: ComponentType<any>;
  export const Scripts: ComponentType<any>;

  export function createFileRoute(path: string): (config: any) => any;
  export function createRootRouteWithContext<T = any>(): (config: any) => any;
  export function createRouter(config: any): any;
  export function useRouter(): any;
}
