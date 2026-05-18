import type { components } from "./types/api";

type S = components["schemas"];

export type JobStatusEnum = S["JobStatusEnum"];
export type FormatTypeEnum = S["FormatTypeEnum"];
export type PlatformEnum = S["PlatformEnum"];
export type AssetTypeEnum = S["AssetTypeEnum"];
export type VerdictEnum = S["VerdictEnum"];

export type ResearchInputs = S["ResearchInputs"];
export type StoryDirectives = S["StoryDirectives"];
export type AssetRenderMeta = S["AssetRenderMeta"];
export type SeoMeta = S["SeoMeta"];

export type VideoScene = S["VideoScene"];
export type VideoFormatPayload = S["VideoFormatPayload"];
export type BlogSection = S["BlogSection"];
export type BlogFormatPayload = S["BlogFormatPayload"];
export type CarouselSlide = S["CarouselSlide"];
export type CarouselFormatPayload = S["CarouselFormatPayload"];

export type JobCreateRequest = S["JobCreateRequest"];
export type ScriptApprovalRequest = S["ScriptApprovalRequest"];
export type FactCheckClaimResponse = S["FactCheckClaimResponse"];
export type ScriptResponse = S["ScriptResponse"];
export type AssetResponse = S["AssetResponse"];
export type RenderJobResponse = S["RenderJobResponse"];
