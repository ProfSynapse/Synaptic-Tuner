"use client"

import { ColumnDef } from "@tanstack/react-table"
import { DatasetEntry } from "@/lib/types"
import { Badge } from "@/components/ui/badge"

export const columns: ColumnDef<DatasetEntry>[] = [
  {
    accessorKey: "index",
    header: "Index",
    cell: ({ row }) => <div className="w-[50px]">{row.index}</div>,
  },
  {
    accessorFn: (row) => {
      const userMsg = row.conversations.find(m => m.role === "user");
      return userMsg?.content || "(No user message)";
    },
    header: "First User Query",
    cell: ({ row }) => {
      const userMsg = row.original.conversations.find(m => m.role === "user");
      return (
        <div className="max-w-[500px] truncate font-medium">
          {userMsg?.content || "(No user message)"}
        </div>
      )
    }
  },
  {
    accessorFn: (row) => {
      return row.conversations.some(m => m.tool_calls && m.tool_calls.length > 0);
    },
    header: "Has Tools",
    cell: ({ row }) => {
      const hasTools = row.original.conversations.some(m => m.tool_calls && m.tool_calls.length > 0);
      return hasTools ? (
        <Badge variant="default">Yes</Badge>
      ) : (
        <Badge variant="secondary">No</Badge>
      )
    }
  },
]
