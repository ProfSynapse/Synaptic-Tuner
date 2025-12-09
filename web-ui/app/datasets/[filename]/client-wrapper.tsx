"use client"

import { useRouter } from "next/navigation";
import { DataTable } from "@/components/data-table";
import { columns } from "./columns";
import { DatasetEntry } from "@/lib/types";

interface ClientWrapperProps {
  data: DatasetEntry[];
  filename: string;
}

export function DatasetTableWrapper({ data, filename }: ClientWrapperProps) {
  const router = useRouter();

  return (
    <DataTable 
      columns={columns} 
      data={data} 
      onRowClick={(row) => {
        // row is the original data object. We need the index.
        // The index is not strictly in the object unless we put it there.
        // But TanStack table knows the index.
        // However, onRowClick in DataTable passes row.original.
        
        // Let's find the index in the original data array
        const index = data.indexOf(row);
        const encodedPath = encodeURIComponent(filename);
        router.push(`/datasets/${encodedPath}/${index}`);
      }}
    />
  );
}
