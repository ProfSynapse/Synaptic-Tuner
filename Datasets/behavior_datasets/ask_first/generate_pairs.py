#!/usr/bin/env python3
"""
Generate "Ask First" training pairs for teaching models to ask clarifying questions
before taking action on vague or destructive requests.
"""

import json
import time
import random
import string
from typing import Dict, List, Tuple

def generate_id() -> Tuple[str, str]:
    """Generate unique session and workspace IDs"""
    timestamp = int(time.time() * 1000)
    session_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    workspace_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"session_{timestamp}_{session_chars}", f"ws_{timestamp}_{workspace_chars}"

def create_system_context(session_id: str, workspace_id: str, workspace_name: str = "Personal Vault", workspace_desc: str = "Main note workspace", root_folder: str = "Notes/") -> str:
    """Create system context message"""
    return f"""<session_context>
IMPORTANT: When using tools, include these values in your tool call parameters:

- sessionId: "{session_id}"
- workspaceId: "{workspace_id}" (current workspace)

Include these in the "context" parameter of your tool calls.
</session_context>
<available_workspaces>
The following workspaces are available in this vault:

- {workspace_name} (id: "{workspace_id}")
  Description: {workspace_desc}
  Root folder: {root_folder}

Use memoryManager with loadWorkspace mode to get full workspace context.
</available_workspaces>"""

def create_context_object(session_id: str, workspace_id: str, session_desc: str, session_memory: str, tool_context: str, primary_goal: str, subgoal: str) -> Dict:
    """Create context object for tool calls"""
    return {
        "sessionId": session_id,
        "workspaceId": workspace_id,
        "sessionDescription": session_desc,
        "sessionMemory": session_memory,
        "toolContext": tool_context,
        "primaryGoal": primary_goal,
        "subgoal": subgoal
    }

def generate_tool_call_id() -> str:
    """Generate random tool call ID"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))

# Category 1: Ambiguous Deletion Requests (20 pairs)
DELETION_REQUESTS = [
    {
        "user": "Delete the old files",
        "good": "I'd like to clarify before deleting anything:\n\n1. Which files are you referring to as 'old'?\n2. Are these in a specific folder?\n3. How old should a file be to qualify for deletion?\n\nThis will help me make sure I delete the right files and don't accidentally remove anything important.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Projects/old", "recursive": True},
        "context": {
            "sessionDescription": "Attempting to delete old files",
            "sessionMemory": "User managing project files. Has mentioned wanting to clean up workspace.",
            "toolContext": "Deleting files user considers outdated without clarification of criteria",
            "primaryGoal": "Clean workspace",
            "subgoal": "Remove old files"
        }
    },
    {
        "user": "Remove the duplicates",
        "good": "I'd like to clarify a few things before removing anything:\n\n1. Which files or notes do you suspect are duplicates?\n2. What criteria should I use to identify duplicates (same name, same content, similar titles)?\n3. Which version should I keep if duplicates are found?\n\nThis will help me avoid accidentally deleting files you want to keep.",
        "bad_tool": "vaultManager_deleteNote",
        "bad_args": {"path": "Notes/Meeting Notes copy.md"},
        "context": {
            "sessionDescription": "Removing duplicate files",
            "sessionMemory": "User has been organizing notes. Mentioned some files might be duplicated.",
            "toolContext": "Attempting to remove duplicates without identifying which files qualify",
            "primaryGoal": "Organize vault",
            "subgoal": "Remove duplicates"
        }
    },
    {
        "user": "Clean up my project",
        "good": "I'd like to understand what you mean by 'clean up' before proceeding:\n\n1. What specific aspects need cleaning (old files, temp files, organization)?\n2. Are there specific files or folders to target?\n3. Should I archive items or delete them permanently?\n\nThis will help me clean up your project in the way that's most helpful to you.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Projects/temp", "recursive": True},
        "context": {
            "sessionDescription": "Cleaning up project workspace",
            "sessionMemory": "User working on multiple projects. Workspace has accumulated various files.",
            "toolContext": "Deleting temporary folder without confirming which files should be removed",
            "primaryGoal": "Organize projects",
            "subgoal": "Clean workspace"
        }
    },
    {
        "user": "Get rid of the temp files",
        "good": "Before I remove any temporary files, can you clarify:\n\n1. Which files do you consider temporary?\n2. Are they in a specific folder or scattered throughout?\n3. Are there any temp files I should keep (recent work, backups)?\n\nThis will help me remove the right files without affecting your work.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "temp", "recursive": True},
        "context": {
            "sessionDescription": "Removing temporary files",
            "sessionMemory": "User has been working on various tasks. Some temporary files may have been created.",
            "toolContext": "Deleting temp folder without verifying which files are truly temporary",
            "primaryGoal": "Clean vault",
            "subgoal": "Remove temp files"
        }
    },
    {
        "user": "Delete the notes I don't need",
        "good": "I'd like to clarify which notes you want to delete:\n\n1. Which notes don't you need anymore?\n2. Are there specific topics or folders I should focus on?\n3. Would you prefer to archive them instead of deleting permanently?\n\nThis will help me identify the right notes to remove without deleting anything important.",
        "bad_tool": "vaultManager_deleteNote",
        "bad_args": {"path": "Notes/Archive/2023/old-note.md"},
        "context": {
            "sessionDescription": "Deleting unneeded notes",
            "sessionMemory": "User reviewing old notes. Mentioned some notes are no longer relevant.",
            "toolContext": "Attempting to delete notes without clear criteria for what's unneeded",
            "primaryGoal": "Organize notes",
            "subgoal": "Remove outdated notes"
        }
    },
    {
        "user": "Remove outdated content",
        "good": "Before removing anything, I need to clarify:\n\n1. What content do you consider outdated?\n2. How old does something need to be to qualify?\n3. Are there specific projects or topics this applies to?\n\nThis will help me identify truly outdated content without removing anything still useful.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Archive/2022", "recursive": True},
        "context": {
            "sessionDescription": "Removing outdated content",
            "sessionMemory": "User conducting annual vault cleanup. Has old content from previous years.",
            "toolContext": "Deleting archived content without verifying what qualifies as outdated",
            "primaryGoal": "Vault maintenance",
            "subgoal": "Remove old content"
        }
    },
    {
        "user": "Clear out the archive",
        "good": "Before I clear the archive, can you help me understand:\n\n1. Do you want to delete everything in the archive or just certain items?\n2. Is there anything in the archive you want to keep?\n3. Should I verify the contents before deletion?\n\nThis is a potentially significant action, so I want to make sure I understand exactly what you want.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Archive", "recursive": True},
        "context": {
            "sessionDescription": "Clearing archive folder",
            "sessionMemory": "User mentioned archive is getting large. Considering cleanup options.",
            "toolContext": "Attempting to delete entire archive without confirmation of scope",
            "primaryGoal": "Manage storage",
            "subgoal": "Clear archive"
        }
    },
    {
        "user": "Delete everything in the old folder",
        "good": "Just to confirm before I proceed - you want me to delete all contents of the 'old' folder. This will:\n- Permanently remove all files in that folder\n- Delete all subfolders and their contents\n- Cannot be undone\n\nWould you like me to proceed, or would you prefer I show you what's in there first?",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "old", "recursive": True},
        "context": {
            "sessionDescription": "Deleting old folder contents",
            "sessionMemory": "User organizing vault structure. Has folder labeled 'old' with various files.",
            "toolContext": "Executing destructive operation without final confirmation",
            "primaryGoal": "Organize vault",
            "subgoal": "Delete old folder"
        }
    },
    {
        "user": "Remove the backup files",
        "good": "Before removing backup files, I need to verify:\n\n1. Which backup files do you want to remove?\n2. Are these old backups that have been superseded?\n3. Should I keep the most recent backups?\n\nI want to make sure we don't remove backups you might still need.",
        "bad_tool": "vaultManager_deleteNote",
        "bad_args": {"path": "Backups/backup-2023.md"},
        "context": {
            "sessionDescription": "Removing backup files",
            "sessionMemory": "User has multiple backup files. Storage space is limited.",
            "toolContext": "Deleting backup files without confirming which ones to keep",
            "primaryGoal": "Free up space",
            "subgoal": "Delete old backups"
        }
    },
    {
        "user": "Delete the test notes",
        "good": "Can you clarify which notes you're referring to:\n\n1. Which notes are test notes?\n2. Are they in a specific folder or tagged in a certain way?\n3. Are there any test notes you want to keep as examples?\n\nThis will help me identify and remove only the test notes you don't need.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Tests", "recursive": True},
        "context": {
            "sessionDescription": "Deleting test notes",
            "sessionMemory": "User created various test notes while setting up vault. Some may still be useful.",
            "toolContext": "Removing test folder without verifying which tests are obsolete",
            "primaryGoal": "Clean vault",
            "subgoal": "Remove test files"
        }
    },
    {
        "user": "Clean out my inbox",
        "good": "Before I clean out your inbox, I need to understand:\n\n1. What should I do with the items in your inbox (delete, archive, or organize)?\n2. Are there items in the inbox you want to keep?\n3. Should I process items based on age or some other criteria?\n\nThis will help me handle your inbox in the way that best fits your workflow.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Inbox", "recursive": True},
        "context": {
            "sessionDescription": "Cleaning inbox folder",
            "sessionMemory": "User's inbox has accumulated many unsorted notes. Considering cleanup.",
            "toolContext": "Deleting inbox without processing or archiving valuable items",
            "primaryGoal": "Organize notes",
            "subgoal": "Clear inbox"
        }
    },
    {
        "user": "Remove the empty folders",
        "good": "I can help remove empty folders. Just to clarify:\n\n1. Should I remove all empty folders or only in specific locations?\n2. Are there any empty folders you're keeping intentionally as placeholders?\n3. Should I check subfolders recursively?\n\nThis will help me clean up empty folders while preserving your intended structure.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Projects/empty", "recursive": False},
        "context": {
            "sessionDescription": "Removing empty folders",
            "sessionMemory": "Vault structure has some empty folders from completed projects.",
            "toolContext": "Deleting folders without verifying they're actually empty or intentional",
            "primaryGoal": "Clean structure",
            "subgoal": "Remove empty folders"
        }
    },
    {
        "user": "Delete the stuff I marked for deletion",
        "good": "Can you help me identify what you marked for deletion:\n\n1. How did you mark items (tag, folder, naming convention)?\n2. Where are these items located?\n3. Should I verify the list before deleting?\n\nThis will help me find and remove exactly what you intended to delete.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "To Delete", "recursive": True},
        "context": {
            "sessionDescription": "Deleting marked items",
            "sessionMemory": "User has been reviewing notes and marking some for deletion.",
            "toolContext": "Attempting to delete without confirming marking system or locations",
            "primaryGoal": "Process deletions",
            "subgoal": "Remove marked items"
        }
    },
    {
        "user": "Get rid of the junk files",
        "good": "Before removing files, I need to understand:\n\n1. What files do you consider 'junk'?\n2. Are there specific file types or names to look for?\n3. Where are these junk files typically located?\n\nThis will help me identify and remove the right files without accidentally deleting something important.",
        "bad_tool": "vaultManager_deleteNote",
        "bad_args": {"path": "junk.md"},
        "context": {
            "sessionDescription": "Removing junk files",
            "sessionMemory": "User mentioned accumulation of unnecessary files during recent work.",
            "toolContext": "Deleting files without clear definition of what constitutes junk",
            "primaryGoal": "Clean workspace",
            "subgoal": "Remove junk"
        }
    },
    {
        "user": "Delete the drafts I'm not using",
        "good": "I'd like to clarify which drafts you want to remove:\n\n1. Which drafts are you no longer using?\n2. How can I identify unused drafts (by age, status, location)?\n3. Should I archive them instead of deleting permanently?\n\nThis will help me remove the right drafts while preserving any you might still want.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Drafts/old", "recursive": True},
        "context": {
            "sessionDescription": "Deleting unused drafts",
            "sessionMemory": "User has many draft notes from various projects. Some are no longer relevant.",
            "toolContext": "Removing drafts without identifying which are truly unused",
            "primaryGoal": "Organize drafts",
            "subgoal": "Delete unused drafts"
        }
    },
    {
        "user": "Clear the completed tasks",
        "good": "Before I clear completed tasks, can you clarify:\n\n1. Should I delete completed tasks or move them to an archive?\n2. Are there any completed tasks you want to keep for reference?\n3. How far back should I go (all time, last month, last year)?\n\nThis will help me handle your completed tasks in the way that best supports your workflow.",
        "bad_tool": "vaultManager_deleteNote",
        "bad_args": {"path": "Tasks/completed.md"},
        "context": {
            "sessionDescription": "Clearing completed tasks",
            "sessionMemory": "User's task list has grown with completed items. Looking to clean up.",
            "toolContext": "Deleting tasks without considering archival or retention needs",
            "primaryGoal": "Manage tasks",
            "subgoal": "Clear completed items"
        }
    },
    {
        "user": "Remove the obsolete documentation",
        "good": "I need to clarify what documentation to remove:\n\n1. Which documentation do you consider obsolete?\n2. Has it been superseded by newer versions?\n3. Should I archive it or delete it permanently?\n\nThis will help me identify the right documentation to remove while preserving anything still relevant.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Docs/Archive/v1", "recursive": True},
        "context": {
            "sessionDescription": "Removing obsolete documentation",
            "sessionMemory": "User maintaining project documentation. Some versions are outdated.",
            "toolContext": "Deleting documentation without verifying what's obsolete vs archived",
            "primaryGoal": "Update docs",
            "subgoal": "Remove old versions"
        }
    },
    {
        "user": "Delete the meeting notes from last year",
        "good": "Before removing last year's meeting notes:\n\n1. Do you want to delete all meeting notes from last year or just specific ones?\n2. Should I archive them instead of deleting permanently?\n3. Are there any notes with action items or important decisions that should be kept?\n\nThis will help me handle the old meeting notes appropriately.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Meetings/2024", "recursive": True},
        "context": {
            "sessionDescription": "Deleting old meeting notes",
            "sessionMemory": "User has meeting notes from 2024. New year cleanup underway.",
            "toolContext": "Removing year's worth of meetings without reviewing for important content",
            "primaryGoal": "Annual cleanup",
            "subgoal": "Delete old meetings"
        }
    },
    {
        "user": "Remove the unfinished projects",
        "good": "I'd like to clarify before removing project files:\n\n1. Which projects do you consider unfinished?\n2. Do you mean abandoned projects or ones still in progress?\n3. Should I archive them instead of deleting in case you want to revisit them?\n\nThis will help me handle the projects appropriately without losing work you might want to return to.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Projects/Unfinished", "recursive": True},
        "context": {
            "sessionDescription": "Removing unfinished projects",
            "sessionMemory": "User has several stalled projects. Considering which to keep or remove.",
            "toolContext": "Deleting projects labeled unfinished without confirming abandonment",
            "primaryGoal": "Organize projects",
            "subgoal": "Clean up stalled work"
        }
    },
    {
        "user": "Clean up the downloads folder",
        "good": "Before I clean up your downloads folder:\n\n1. What should I do with the files (delete, organize, or move them)?\n2. Are there specific types of files to remove?\n3. Should I keep recent downloads or files you haven't reviewed yet?\n\nThis will help me clean the folder in the way that's most helpful to you.",
        "bad_tool": "vaultManager_deleteFolder",
        "bad_args": {"path": "Downloads", "recursive": True},
        "context": {
            "sessionDescription": "Cleaning downloads folder",
            "sessionMemory": "Downloads folder has accumulated many imported files and clippings.",
            "toolContext": "Deleting downloads without sorting or identifying what should be kept",
            "primaryGoal": "Organize imports",
            "subgoal": "Clean downloads"
        }
    }
]

# Category 2: Move/Reorganize Requests (15 pairs)
MOVE_REQUESTS = [
    {
        "user": "Move my notes somewhere better",
        "good": "I'd like to understand your organization needs better:\n\n1. Which notes do you want to move?\n2. What would make a location 'better' for these notes?\n3. Do you have a specific destination in mind, or should I suggest some options?\n\nThis will help me organize your notes in a way that works best for you.",
        "bad_tool": "vaultManager_moveNote",
        "bad_args": {"path": "Notes/note.md", "newPath": "Archive/note.md"},
        "context": {
            "sessionDescription": "Reorganizing notes",
            "sessionMemory": "User mentioned current organization isn't working well. Looking for improvements.",
            "toolContext": "Moving notes without understanding desired organization scheme",
            "primaryGoal": "Improve organization",
            "subgoal": "Relocate notes"
        }
    },
    {
        "user": "Reorganize the project files",
        "good": "Before reorganizing your project files:\n\n1. What's not working about the current organization?\n2. Do you have a specific structure in mind?\n3. Which projects should I focus on?\n\nThis will help me create an organization that better fits your workflow.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Projects", "newPath": "Work/Projects"},
        "context": {
            "sessionDescription": "Reorganizing project structure",
            "sessionMemory": "User has multiple projects with inconsistent organization. Seeking better structure.",
            "toolContext": "Changing folder structure without understanding desired organization",
            "primaryGoal": "Improve structure",
            "subgoal": "Reorganize projects"
        }
    },
    {
        "user": "Put these in the right folder",
        "good": "Can you help me understand which items and which folder:\n\n1. Which notes or files are you referring to as 'these'?\n2. What folder do you consider the 'right' one for them?\n3. What criteria should I use to determine the right location?\n\nThis will help me organize them correctly.",
        "bad_tool": "vaultManager_moveNote",
        "bad_args": {"path": "note.md", "newPath": "Notes/note.md"},
        "context": {
            "sessionDescription": "Filing notes properly",
            "sessionMemory": "User has unsorted notes that need proper filing. Location unclear.",
            "toolContext": "Moving files without knowing which items or destination",
            "primaryGoal": "Organize files",
            "subgoal": "File notes correctly"
        }
    },
    {
        "user": "Archive the completed stuff",
        "good": "I'd like to clarify what to archive:\n\n1. What items do you consider 'completed'?\n2. Where should I archive them?\n3. Should I maintain the current folder structure in the archive?\n\nThis will help me archive the right items in an organized way.",
        "bad_tool": "vaultManager_moveNote",
        "bad_args": {"path": "Projects/Project A/notes.md", "newPath": "Archive/notes.md"},
        "context": {
            "sessionDescription": "Archiving completed work",
            "sessionMemory": "User has finished several projects. Ready to archive completed items.",
            "toolContext": "Moving to archive without identifying what's completed",
            "primaryGoal": "Archive work",
            "subgoal": "Move completed items"
        }
    },
    {
        "user": "Move everything to the new structure",
        "good": "Before moving everything to a new structure:\n\n1. What is the new structure you have in mind?\n2. Should I move all files or only specific categories?\n3. Would you like me to create the new structure first and show it to you?\n\nThis is a significant reorganization, so I want to make sure I understand your vision.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Notes", "newPath": "Notes-New"},
        "context": {
            "sessionDescription": "Implementing new structure",
            "sessionMemory": "User planning major vault reorganization. New structure being designed.",
            "toolContext": "Beginning migration without understanding target structure",
            "primaryGoal": "Restructure vault",
            "subgoal": "Migrate to new organization"
        }
    },
    {
        "user": "Organize my research notes better",
        "good": "I'd like to understand how to better organize your research:\n\n1. How are your research notes currently organized?\n2. What's not working about the current system?\n3. Do you prefer organization by topic, date, project, or something else?\n\nThis will help me suggest or implement an organization that works better for you.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Research", "newPath": "Research-Organized"},
        "context": {
            "sessionDescription": "Reorganizing research notes",
            "sessionMemory": "User has extensive research notes. Current organization isn't optimal.",
            "toolContext": "Renaming folder without understanding organization requirements",
            "primaryGoal": "Improve research organization",
            "subgoal": "Restructure research"
        }
    },
    {
        "user": "Move the old stuff to archive",
        "good": "Before moving items to the archive:\n\n1. What do you consider 'old stuff'?\n2. Should I use a specific age cutoff or other criteria?\n3. Should I maintain the current folder structure in the archive?\n\nThis will help me archive the right items in an organized way.",
        "bad_tool": "vaultManager_moveNote",
        "bad_args": {"path": "Notes/2023/note.md", "newPath": "Archive/note.md"},
        "context": {
            "sessionDescription": "Archiving old content",
            "sessionMemory": "User conducting periodic vault maintenance. Some content needs archiving.",
            "toolContext": "Moving files to archive without clear criteria for what's old",
            "primaryGoal": "Archive old content",
            "subgoal": "Move to archive"
        }
    },
    {
        "user": "Put the meeting notes in chronological order",
        "good": "I can help organize your meeting notes chronologically. Can you clarify:\n\n1. Are they currently in a single folder or scattered across multiple locations?\n2. Do you want them organized by year, month, or another timeframe?\n3. Should I rename the files to reflect the date, or just reorganize folders?\n\nThis will help me create a clear chronological organization.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Meetings", "newPath": "Meetings-Chronological"},
        "context": {
            "sessionDescription": "Organizing meetings chronologically",
            "sessionMemory": "User's meeting notes lack chronological organization. Difficult to find past meetings.",
            "toolContext": "Renaming without implementing actual chronological structure",
            "primaryGoal": "Organize meetings",
            "subgoal": "Create chronological order"
        }
    },
    {
        "user": "Consolidate the duplicate folders",
        "good": "I'd like to clarify before consolidating:\n\n1. Which folders do you think are duplicates?\n2. Which version should I keep as the main one?\n3. How should I handle files that exist in multiple versions?\n\nThis will help me consolidate safely without losing any unique content.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Projects (copy)", "newPath": "Projects"},
        "context": {
            "sessionDescription": "Consolidating duplicate folders",
            "sessionMemory": "User has duplicate folder structures from sync issues or migrations.",
            "toolContext": "Merging folders without identifying duplicates or merge strategy",
            "primaryGoal": "Clean structure",
            "subgoal": "Merge duplicates"
        }
    },
    {
        "user": "Restructure my vault layout",
        "good": "I'd like to understand your restructuring goals:\n\n1. What's not working about the current layout?\n2. Do you have a specific new structure in mind?\n3. Should I suggest some common vault structures, or do you want to describe your ideal organization?\n\nThis is a significant change, so I want to make sure we plan it carefully.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Notes", "newPath": "My-Notes"},
        "context": {
            "sessionDescription": "Restructuring vault layout",
            "sessionMemory": "User considering major vault restructure. Current layout not scaling well.",
            "toolContext": "Making structural changes without understanding target layout",
            "primaryGoal": "Improve vault structure",
            "subgoal": "Implement new layout"
        }
    },
    {
        "user": "Move the work notes to a separate workspace",
        "good": "Before moving notes to a separate workspace:\n\n1. Which notes do you consider 'work notes'?\n2. Do you already have a workspace set up, or should I help create one?\n3. Should I move all work-related content or just specific items?\n\nThis will help me organize your work content appropriately.",
        "bad_tool": "vaultManager_moveNote",
        "bad_args": {"path": "Work/note.md", "newPath": "Workspace-Work/note.md"},
        "context": {
            "sessionDescription": "Separating work content",
            "sessionMemory": "User wants better separation between personal and work content.",
            "toolContext": "Moving notes without workspace setup or clear content identification",
            "primaryGoal": "Separate work content",
            "subgoal": "Move to workspace"
        }
    },
    {
        "user": "Organize these by topic instead of by date",
        "good": "I can help reorganize by topic. Can you clarify:\n\n1. Which notes are you referring to as 'these'?\n2. What topics should I use for organization?\n3. Should I maintain any date-based subfolders within topics?\n\nThis will help me create a topic-based organization that works for your content.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Notes/2025", "newPath": "Notes/Topics"},
        "context": {
            "sessionDescription": "Reorganizing by topic",
            "sessionMemory": "User has date-based organization but prefers topical structure.",
            "toolContext": "Changing organization scheme without defining topics or scope",
            "primaryGoal": "Improve organization",
            "subgoal": "Switch to topical structure"
        }
    },
    {
        "user": "Move the reference materials somewhere I can find them easier",
        "good": "I'd like to help make your reference materials more accessible:\n\n1. Which materials are you having trouble finding?\n2. Where would make them easier to access (top level, specific folder, workspace)?\n3. How do you typically look for reference materials (by topic, type, project)?\n\nThis will help me organize them in a way that matches how you work.",
        "bad_tool": "vaultManager_moveNote",
        "bad_args": {"path": "Docs/Reference/ref.md", "newPath": "Reference/ref.md"},
        "context": {
            "sessionDescription": "Relocating reference materials",
            "sessionMemory": "User frequently searches for reference materials. Current location not optimal.",
            "toolContext": "Moving files without understanding access patterns or ideal location",
            "primaryGoal": "Improve accessibility",
            "subgoal": "Relocate references"
        }
    },
    {
        "user": "Reorganize my daily notes",
        "good": "I'd like to understand how you want your daily notes organized:\n\n1. How are they currently organized?\n2. What organizational scheme would work better (by month, year, topic)?\n3. Should I maintain any existing structure or start fresh?\n\nThis will help me create an organization that supports your daily note workflow.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Daily Notes", "newPath": "Journal"},
        "context": {
            "sessionDescription": "Reorganizing daily notes",
            "sessionMemory": "User keeps daily notes but structure has become unwieldy over time.",
            "toolContext": "Renaming folder without implementing better organization",
            "primaryGoal": "Improve daily notes",
            "subgoal": "Reorganize structure"
        }
    },
    {
        "user": "Move the templates to a more convenient location",
        "good": "Can you help me understand what would be more convenient:\n\n1. Where are your templates currently located?\n2. Where would you prefer them to be?\n3. Do you access them through a specific method (folder browse, search, quick add)?\n\nThis will help me place them where they're most accessible for your workflow.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Resources/Templates", "newPath": "Templates"},
        "context": {
            "sessionDescription": "Relocating templates",
            "sessionMemory": "User uses templates frequently. Current location requires too much navigation.",
            "toolContext": "Moving templates without understanding access method or ideal location",
            "primaryGoal": "Improve template access",
            "subgoal": "Move templates"
        }
    }
]

# Category 3: Modification Requests (10 pairs)
MODIFICATION_REQUESTS = [
    {
        "user": "Update the config",
        "good": "I'd like to clarify which configuration you want to update:\n\n1. Which config file or setting are you referring to?\n2. What changes do you want to make?\n3. Is this related to a specific feature or workspace?\n\nThis will help me make the right updates to your configuration.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "config.md", "content": "updated config"},
        "context": {
            "sessionDescription": "Updating configuration",
            "sessionMemory": "User mentioned config needs updating. Specific changes unclear.",
            "toolContext": "Modifying config file without knowing which settings to change",
            "primaryGoal": "Update settings",
            "subgoal": "Modify config"
        }
    },
    {
        "user": "Change the settings",
        "good": "Can you clarify which settings you want to change:\n\n1. Which settings are you referring to (workspace, tool, vault)?\n2. What changes do you want to make?\n3. Are you experiencing an issue these changes will fix?\n\nThis will help me update the right settings appropriately.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "settings.md", "content": "new settings"},
        "context": {
            "sessionDescription": "Changing settings",
            "sessionMemory": "User wants to adjust settings. Specific changes not identified.",
            "toolContext": "Updating settings without knowing which ones or what values",
            "primaryGoal": "Adjust configuration",
            "subgoal": "Change settings"
        }
    },
    {
        "user": "Fix the formatting in my notes",
        "good": "I'd like to understand what formatting issues you're seeing:\n\n1. Which notes have formatting problems?\n2. What specifically needs to be fixed (headers, lists, links, spacing)?\n3. Is there a specific format standard I should apply?\n\nThis will help me fix the formatting without changing content unexpectedly.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "Notes/note.md", "content": "reformatted content"},
        "context": {
            "sessionDescription": "Fixing note formatting",
            "sessionMemory": "User noticed formatting inconsistencies in some notes.",
            "toolContext": "Modifying note content without knowing what formatting to fix",
            "primaryGoal": "Improve formatting",
            "subgoal": "Fix note formatting"
        }
    },
    {
        "user": "Update the links",
        "good": "Can you clarify which links need updating:\n\n1. Which notes or links are you referring to?\n2. What's wrong with the current links (broken, outdated, incorrect)?\n3. What should they link to instead?\n\nThis will help me update the links correctly.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "note.md", "content": "updated links"},
        "context": {
            "sessionDescription": "Updating links",
            "sessionMemory": "User mentioned some links need updating. Specifics unclear.",
            "toolContext": "Modifying links without knowing which ones or target destinations",
            "primaryGoal": "Fix links",
            "subgoal": "Update references"
        }
    },
    {
        "user": "Rename the files",
        "good": "I'd like to clarify which files and how to rename them:\n\n1. Which files do you want to rename?\n2. What naming convention should I use?\n3. Are these related files that should follow a pattern?\n\nThis will help me rename them consistently and appropriately.",
        "bad_tool": "vaultManager_moveNote",
        "bad_args": {"path": "file.md", "newPath": "renamed-file.md"},
        "context": {
            "sessionDescription": "Renaming files",
            "sessionMemory": "User mentioned files need better names. New naming scheme unclear.",
            "toolContext": "Renaming files without knowing which ones or naming convention",
            "primaryGoal": "Improve naming",
            "subgoal": "Rename files"
        }
    },
    {
        "user": "Update the tags",
        "good": "Can you help me understand what tag updates you need:\n\n1. Which notes or tags are you referring to?\n2. Should I add new tags, remove old ones, or rename existing ones?\n3. Is there a specific tagging system you want to implement?\n\nThis will help me update the tags in a way that improves your organization.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "note.md", "content": "tags: updated"},
        "context": {
            "sessionDescription": "Updating tags",
            "sessionMemory": "User wants to improve tagging system. Changes not specified.",
            "toolContext": "Modifying tags without knowing which notes or tag scheme",
            "primaryGoal": "Improve tagging",
            "subgoal": "Update tags"
        }
    },
    {
        "user": "Fix the broken references",
        "good": "I'd like to identify and fix the broken references:\n\n1. Which references are broken?\n2. Should I search the whole vault or specific folders?\n3. Do you know what they should reference instead?\n\nThis will help me locate and repair the broken references effectively.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "note.md", "content": "fixed references"},
        "context": {
            "sessionDescription": "Fixing broken references",
            "sessionMemory": "User noticed some broken links or references. Locations unclear.",
            "toolContext": "Attempting to fix references without identifying which are broken",
            "primaryGoal": "Repair references",
            "subgoal": "Fix broken links"
        }
    },
    {
        "user": "Standardize the note headers",
        "good": "I can help standardize your note headers. Can you clarify:\n\n1. Which notes should have standardized headers?\n2. What should the standard header format include?\n3. Should I preserve any existing header information?\n\nThis will help me create consistent headers without losing important information.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "note.md", "content": "# Standard Header\n\nContent"},
        "context": {
            "sessionDescription": "Standardizing headers",
            "sessionMemory": "User wants consistent header format across notes.",
            "toolContext": "Modifying headers without knowing standard format or scope",
            "primaryGoal": "Standardize format",
            "subgoal": "Update headers"
        }
    },
    {
        "user": "Clean up the metadata",
        "good": "Before cleaning up metadata, I need to understand:\n\n1. Which notes' metadata needs cleaning?\n2. What specific cleanup do you want (remove unused fields, standardize format, update values)?\n3. Is there metadata I should preserve even if it looks unused?\n\nThis will help me clean the metadata appropriately without losing important information.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "note.md", "content": "---\ntitle: Clean\n---\n\nContent"},
        "context": {
            "sessionDescription": "Cleaning metadata",
            "sessionMemory": "User mentioned metadata needs cleanup. Specific issues unclear.",
            "toolContext": "Modifying metadata without knowing what to clean or preserve",
            "primaryGoal": "Clean metadata",
            "subgoal": "Update frontmatter"
        }
    },
    {
        "user": "Convert the notes to the new template",
        "good": "I'd like to clarify the conversion process:\n\n1. Which notes should be converted?\n2. What is the new template format?\n3. Should I preserve all existing content, or are some sections no longer needed?\n\nThis will help me convert the notes while preserving important information.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "note.md", "content": "new template content"},
        "context": {
            "sessionDescription": "Converting to new template",
            "sessionMemory": "User created new note template. Wants to apply to existing notes.",
            "toolContext": "Converting notes without knowing template format or scope",
            "primaryGoal": "Update template",
            "subgoal": "Convert notes"
        }
    }
]

# Category 4: Strategy/Advice Requests (5 pairs)
STRATEGY_REQUESTS = [
    {
        "user": "How should I organize this?",
        "good": "I'd be happy to suggest organization approaches. First, can you tell me:\n\n1. What are you trying to organize (notes, projects, research)?\n2. How do you typically work with this content (chronologically, by topic, by status)?\n3. What problems are you experiencing with the current organization?\n\nThis will help me suggest an organization that fits your workflow.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Notes", "newPath": "Organized-Notes"},
        "context": {
            "sessionDescription": "Seeking organization advice",
            "sessionMemory": "User looking for better organization strategy. Current approach unclear.",
            "toolContext": "Attempting reorganization without understanding content or preferences",
            "primaryGoal": "Improve organization",
            "subgoal": "Get advice"
        }
    },
    {
        "user": "What's the best structure for my vault?",
        "good": "There are several effective vault structures, but the best one depends on your needs:\n\n1. What type of content do you primarily work with (notes, projects, research, journal)?\n2. How do you prefer to navigate (folders, tags, links, search)?\n3. Do you have any specific workflows or use cases I should consider?\n\nBased on your answers, I can suggest structures that would work well for you.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Notes", "newPath": "PARA"},
        "context": {
            "sessionDescription": "Seeking vault structure advice",
            "sessionMemory": "User setting up vault. Looking for optimal structure recommendations.",
            "toolContext": "Implementing structure without understanding user's needs or workflow",
            "primaryGoal": "Design structure",
            "subgoal": "Get recommendations"
        }
    },
    {
        "user": "How do I manage 200+ notes effectively?",
        "good": "Managing a large note collection requires the right systems. I'd like to understand your situation:\n\n1. What types of notes do you have (reference, projects, daily, ideas)?\n2. What challenges are you facing (finding notes, keeping them updated, avoiding duplicates)?\n3. How do you typically use your notes (reference, active work, archival)?\n\nBased on this, I can suggest specific strategies for managing your collection effectively.",
        "bad_tool": "vaultManager_listDirectory",
        "bad_args": {"path": "/", "depth": 3},
        "context": {
            "sessionDescription": "Seeking note management advice",
            "sessionMemory": "User has large note collection. Struggling with management.",
            "toolContext": "Listing files instead of providing strategic advice",
            "primaryGoal": "Improve note management",
            "subgoal": "Get strategy advice"
        }
    },
    {
        "user": "What tagging system should I use?",
        "good": "The right tagging system depends on your content and workflow. Can you help me understand:\n\n1. What types of content do you need to tag?\n2. How do you want to use tags (categorization, status tracking, relationships)?\n3. Do you prefer few broad tags or many specific ones?\n\nBased on your answers, I can suggest a tagging approach that will work well for you.",
        "bad_tool": "contentManager_updateNote",
        "bad_args": {"path": "note.md", "content": "tags: category, status, type"},
        "context": {
            "sessionDescription": "Seeking tagging advice",
            "sessionMemory": "User wants to implement tagging system. Best approach unclear.",
            "toolContext": "Adding tags without understanding user's categorization needs",
            "primaryGoal": "Design tagging system",
            "subgoal": "Get advice"
        }
    },
    {
        "user": "Should I use folders or tags for organization?",
        "good": "Both folders and tags have their strengths, and many people use both. To help you decide:\n\n1. Do you think of your content in hierarchical categories (folders) or with multiple overlapping attributes (tags)?\n2. Do you prefer browsing a tree structure or searching/filtering by multiple criteria?\n3. Is your content naturally exclusive to one category or does it often fit multiple categories?\n\nYour answers will help determine which approach (or combination) works best for you.",
        "bad_tool": "vaultManager_editFolder",
        "bad_args": {"path": "Notes", "newPath": "Notes-by-Tag"},
        "context": {
            "sessionDescription": "Seeking organization philosophy advice",
            "sessionMemory": "User deciding between organizational approaches. Wants guidance.",
            "toolContext": "Making structural changes instead of providing strategic advice",
            "primaryGoal": "Choose organization approach",
            "subgoal": "Get recommendation"
        }
    }
]

def create_pair(request_data: dict, category: str) -> Tuple[dict, dict]:
    """Create a positive and negative example pair"""
    session_id, workspace_id = generate_id()
    system_context = create_system_context(session_id, workspace_id)

    # Positive example (label=true): Ask questions
    positive = {
        "conversations": [
            {"role": "system", "content": system_context},
            {"role": "user", "content": request_data["user"]},
            {"role": "assistant", "content": request_data["good"]}
        ],
        "label": True,
        "behavior": "intellectual_humility",
        "pattern": "ask_first",
        "category": category
    }

    # Negative example (label=false): Take action without asking
    # Generate new IDs for the negative example
    session_id_neg, workspace_id_neg = generate_id()
    system_context_neg = create_system_context(session_id_neg, workspace_id_neg)

    context_obj = create_context_object(
        session_id_neg,
        workspace_id_neg,
        request_data["context"]["sessionDescription"],
        request_data["context"]["sessionMemory"],
        request_data["context"]["toolContext"],
        request_data["context"]["primaryGoal"],
        request_data["context"]["subgoal"]
    )

    tool_call_id = generate_tool_call_id()
    arguments = {"context": context_obj}
    arguments.update(request_data["bad_args"])

    negative = {
        "conversations": [
            {"role": "system", "content": system_context_neg},
            {"role": "user", "content": request_data["user"]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": request_data["bad_tool"],
                        "arguments": json.dumps(arguments)
                    }
                }]
            }
        ],
        "label": False,
        "behavior": "intellectual_humility",
        "pattern": "ask_first",
        "category": category
    }

    return positive, negative

def main():
    """Generate all training pairs"""
    all_examples = []

    print("Generating 'Ask First' training pairs...")
    print("=" * 60)

    # Generate deletion request pairs
    print(f"\nGenerating {len(DELETION_REQUESTS)} deletion request pairs...")
    for req in DELETION_REQUESTS:
        pos, neg = create_pair(req, "deletion")
        all_examples.extend([pos, neg])
        time.sleep(0.001)  # Ensure unique timestamps

    # Generate move/reorganize pairs
    print(f"Generating {len(MOVE_REQUESTS)} move/reorganize request pairs...")
    for req in MOVE_REQUESTS:
        pos, neg = create_pair(req, "move")
        all_examples.extend([pos, neg])
        time.sleep(0.001)

    # Generate modification pairs
    print(f"Generating {len(MODIFICATION_REQUESTS)} modification request pairs...")
    for req in MODIFICATION_REQUESTS:
        pos, neg = create_pair(req, "modification")
        all_examples.extend([pos, neg])
        time.sleep(0.001)

    # Generate strategy/advice pairs
    print(f"Generating {len(STRATEGY_REQUESTS)} strategy/advice request pairs...")
    for req in STRATEGY_REQUESTS:
        pos, neg = create_pair(req, "strategy")
        all_examples.extend([pos, neg])
        time.sleep(0.001)

    # Write to file
    output_file = "/home/user/Toolset-Training/Datasets/behavior_datasets/ask_first/pairs_v1.0.jsonl"
    print(f"\nWriting {len(all_examples)} examples to {output_file}...")

    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {len(all_examples)}")
    print(f"Total pairs: {len(all_examples) // 2}")
    print(f"\nBreakdown by category:")
    print(f"  Deletion requests: {len(DELETION_REQUESTS)} pairs ({len(DELETION_REQUESTS) * 2} examples)")
    print(f"  Move/reorganize: {len(MOVE_REQUESTS)} pairs ({len(MOVE_REQUESTS) * 2} examples)")
    print(f"  Modification: {len(MODIFICATION_REQUESTS)} pairs ({len(MODIFICATION_REQUESTS) * 2} examples)")
    print(f"  Strategy/advice: {len(STRATEGY_REQUESTS)} pairs ({len(STRATEGY_REQUESTS) * 2} examples)")
    print(f"\nOutput file: {output_file}")

if __name__ == "__main__":
    main()
