#!/usr/bin/env python3
"""
Convert Obsidian vault chapters to ChatML training format.

Usage:
    python vault_to_chatml.py <input_file> [--output <output_file>] [--preview]
    python vault_to_chatml.py <input_dir> --recursive [--output <output_file>]

Examples:
    python vault_to_chatml.py "/path/to/chapter.md" --preview
    python vault_to_chatml.py "/path/to/book/" --recursive --output dataset.jsonl
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata, remaining_content)."""
    if not content.startswith('---'):
        return {}, content

    lines = content.split('\n')
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == '---':
            end_idx = i
            break

    if end_idx is None:
        return {}, content

    frontmatter = {}
    current_key = None
    current_list = None

    for line in lines[1:end_idx]:
        if not line.strip():
            continue

        if line.strip().startswith('- '):
            if current_key and current_list is not None:
                value = line.strip()[2:].strip().strip('"').strip("'")
                value = re.sub(r'\[\[([^\]|]+)(\|[^\]]+)?\]\]', r'\1', value)
                current_list.append(value)
            continue

        if ':' in line and not line.startswith(' '):
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ''
            value = re.sub(r'\[\[([^\]|]+)(\|[^\]]+)?\]\]', r'\1', value)
            value = value.strip('"').strip("'")

            if value:
                frontmatter[key] = value
                current_key = None
                current_list = None
            else:
                current_key = key
                current_list = []
                frontmatter[key] = current_list

    remaining = '\n'.join(lines[end_idx + 1:])
    return frontmatter, remaining


def strip_yaml_blocks(content: str) -> str:
    """Remove ```yaml ... ``` code blocks."""
    return re.sub(r'```yaml\n.*?```', '', content, flags=re.DOTALL)


def strip_image_embeds(content: str) -> str:
    """Remove ![[image]] embeds."""
    return re.sub(r'!\[\[[^\]]+\]\]', '', content)


def strip_widget_blocks(content: str) -> str:
    """Remove ```widget ... ``` blocks."""
    return re.sub(r'```widget\n.*?```', '', content, flags=re.DOTALL)


def strip_callouts(content: str) -> str:
    """Remove >[!type] callout blocks."""
    lines = content.split('\n')
    result = []
    in_callout = False

    for line in lines:
        if line.strip().startswith('>[!'):
            in_callout = True
            continue
        if in_callout:
            if line.strip().startswith('>'):
                continue
            else:
                in_callout = False
        if not in_callout:
            result.append(line)

    return '\n'.join(result)


def strip_iframes(content: str) -> str:
    """Remove <iframe> tags."""
    return re.sub(r'<iframe[^>]*>.*?</iframe>', '', content, flags=re.DOTALL)


def clean_prose(content: str) -> str:
    """Clean content to extract just prose."""
    content = strip_yaml_blocks(content)
    content = strip_image_embeds(content)
    content = strip_widget_blocks(content)
    content = strip_callouts(content)
    content = strip_iframes(content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def split_into_scenes(content: str, divider: str = '---') -> list[str]:
    """Split content into scenes based on divider."""
    scenes = re.split(rf'\n{divider}\n', content)
    cleaned = []
    for scene in scenes:
        scene = scene.strip()
        if scene and len(scene) > 50:
            cleaned.append(scene)
    return cleaned


def get_last_paragraph(text: str, max_words: int = 50) -> str:
    """Get the last paragraph or last N words."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return ""

    last = paragraphs[-1]
    words = last.split()

    if len(words) <= max_words:
        return last
    return '...' + ' '.join(words[-max_words:])


def format_entities(frontmatter: dict, style: str = 'list') -> str:
    """Format entities from frontmatter in different styles."""
    entities = {}
    for key in ['characters', 'locations', 'objects', 'groups', 'ships']:
        if key in frontmatter and frontmatter[key]:
            items = frontmatter[key]
            if isinstance(items, list) and items:
                entities[key] = items

    if not entities:
        return ""

    if style == 'list':
        lines = []
        for key, items in entities.items():
            lines.append(f"- {key.capitalize()}: {', '.join(items)}")
        return '\n'.join(lines)
    elif style == 'prose':
        parts = []
        if 'characters' in entities:
            parts.append(f"featuring {', '.join(entities['characters'])}")
        if 'locations' in entities:
            parts.append(f"set in {', '.join(entities['locations'])}")
        if 'objects' in entities:
            parts.append(f"involving {', '.join(entities['objects'])}")
        return ', '.join(parts)
    elif style == 'brief':
        all_items = []
        for items in entities.values():
            all_items.extend(items[:2])  # Just first 2 of each
        return ', '.join(all_items[:5])  # Max 5 total
    else:
        return ""


def get_scene_position_text(scene_index: int, total_scenes: int, style: str = 'formal') -> str:
    """Get scene position text in different styles."""
    if total_scenes <= 1:
        return ""

    is_opening = scene_index == 0
    is_closing = scene_index == total_scenes - 1

    if style == 'formal':
        if is_opening:
            return "This is the opening scene of the chapter."
        elif is_closing:
            return "This is the concluding scene of the chapter."
        else:
            return f"This is scene {scene_index + 1} of {total_scenes}."
    elif style == 'casual':
        if is_opening:
            return "Start from the beginning of this chapter."
        elif is_closing:
            return "This wraps up the chapter."
        else:
            return f"We're in the middle of the chapter (scene {scene_index + 1}/{total_scenes})."
    elif style == 'minimal':
        if is_opening:
            return "Opening scene."
        elif is_closing:
            return "Final scene."
        else:
            return f"Scene {scene_index + 1}/{total_scenes}."
    elif style == 'none':
        return ""
    return ""


def get_continuation_text(previous_scene: Optional[str], style: str = 'quote') -> str:
    """Get continuation text in different styles."""
    if not previous_scene:
        return ""

    last_para = get_last_paragraph(previous_scene)
    if not last_para:
        return ""

    if style == 'quote':
        return f"Continue from:\n> {last_para}"
    elif style == 'inline':
        # Shorter excerpt
        words = last_para.split()
        short = ' '.join(words[-25:]) if len(words) > 25 else last_para
        return f"Pick up from: \"{short}\""
    elif style == 'directive':
        words = last_para.split()
        short = ' '.join(words[-20:]) if len(words) > 20 else last_para
        return f"The previous scene ended with: \"{short}\" Continue from there."
    elif style == 'minimal':
        words = last_para.split()
        short = ' '.join(words[-15:]) if len(words) > 15 else last_para
        return f"[Continuing from: ...{short}]"
    return ""


# Prompt templates - each returns a formatted prompt string
PROMPT_TEMPLATES = [
    # Template 1: Formal, comprehensive
    lambda ctx: "\n\n".join(filter(None, [
        f"Help me draft chapter \"{ctx['chapter']}\" of my book \"{ctx['book']}\" from the point of view of {ctx['pov']}.",
        f"In this chapter: {ctx['description']}" if ctx['description'] else None,
        f"Try to include the following:\n{ctx['entities_list']}" if ctx['entities_list'] else None,
        ctx['position_formal'],
        ctx['continue_quote'],
    ])),

    # Template 2: Casual, conversational
    lambda ctx: "\n\n".join(filter(None, [
        f"I'm working on \"{ctx['book']}\" and need help with chapter \"{ctx['chapter']}\". It's from {ctx['pov']}'s perspective.",
        f"Here's what happens: {ctx['description']}" if ctx['description'] else None,
        f"Key elements to weave in: {ctx['entities_brief']}" if ctx['entities_brief'] else None,
        ctx['position_casual'],
        ctx['continue_inline'],
    ])),

    # Template 3: Direct and minimal
    lambda ctx: "\n\n".join(filter(None, [
        f"Write a scene for \"{ctx['book']}\" - {ctx['chapter']}.",
        f"POV: {ctx['pov']}",
        f"Summary: {ctx['description']}" if ctx['description'] else None,
        ctx['position_minimal'],
        ctx['continue_minimal'],
    ])),

    # Template 4: Story-focused
    lambda ctx: "\n\n".join(filter(None, [
        f"Continue my novel \"{ctx['book']}\".",
        f"Chapter: {ctx['chapter']} | Narrator: {ctx['pov']}",
        f"{ctx['description']}" if ctx['description'] else None,
        f"Elements: {ctx['entities_prose']}" if ctx['entities_prose'] else None,
        ctx['continue_directive'],
    ])),

    # Template 5: Collaborative tone
    lambda ctx: "\n\n".join(filter(None, [
        f"Can you help me write the next scene in my book?",
        f"Book: \"{ctx['book']}\"\nChapter: \"{ctx['chapter']}\"\nPOV: {ctx['pov']}",
        f"What's happening: {ctx['description']}" if ctx['description'] else None,
        f"Include if possible:\n{ctx['entities_list']}" if ctx['entities_list'] else None,
        ctx['position_casual'],
        ctx['continue_quote'],
    ])),

    # Template 6: Brief request
    lambda ctx: "\n\n".join(filter(None, [
        f"Draft a scene: \"{ctx['chapter']}\" from \"{ctx['book']}\"",
        f"({ctx['pov']}'s POV) - {ctx['description']}" if ctx['description'] else f"({ctx['pov']}'s POV)",
        ctx['continue_inline'] or ctx['position_minimal'],
    ])),

    # Template 7: Detailed worldbuilding focus
    lambda ctx: "\n\n".join(filter(None, [
        f"I need help drafting a scene for my novel \"{ctx['book']}\".",
        f"This is chapter \"{ctx['chapter']}\", told from {ctx['pov']}'s point of view.",
        f"Scene description: {ctx['description']}" if ctx['description'] else None,
        f"The scene should feature {ctx['entities_prose']}." if ctx['entities_prose'] else None,
        ctx['position_formal'],
        ctx['continue_quote'],
    ])),

    # Template 8: Action-oriented
    lambda ctx: "\n\n".join(filter(None, [
        f"Write the next part of \"{ctx['book']}\" for me.",
        f"Chapter \"{ctx['chapter']}\" - {ctx['pov']}'s perspective.",
        f"{ctx['description']}" if ctx['description'] else None,
        ctx['position_casual'],
        ctx['continue_directive'],
    ])),

    # Template 9: Question-style
    lambda ctx: "\n\n".join(filter(None, [
        f"What happens next in \"{ctx['book']}\"?",
        f"I'm writing chapter \"{ctx['chapter']}\" from {ctx['pov']}'s POV.",
        f"The scene: {ctx['description']}" if ctx['description'] else None,
        ctx['continue_quote'],
    ])),

    # Template 10: Workshop/outline style
    lambda ctx: "\n\n".join(filter(None, [
        f"Scene outline:",
        f"- Novel: {ctx['book']}",
        f"- Chapter: {ctx['chapter']}",
        f"- POV: {ctx['pov']}",
        f"- Summary: {ctx['description']}" if ctx['description'] else None,
        f"- Elements: {ctx['entities_brief']}" if ctx['entities_brief'] else None,
        "",
        "Write this scene." + (f" {ctx['position_minimal']}" if ctx['position_minimal'] else ""),
        ctx['continue_minimal'],
    ])),

    # Template 11: Character-focused
    lambda ctx: "\n\n".join(filter(None, [
        f"Write a scene from {ctx['pov']}'s perspective.",
        f"This is from \"{ctx['book']}\", chapter \"{ctx['chapter']}\".",
        f"{ctx['description']}" if ctx['description'] else None,
        ctx['position_casual'],
        ctx['continue_inline'],
    ])),

    # Template 12: Cowriter energy
    lambda ctx: "\n\n".join(filter(None, [
        f"Hey, I need your help with my novel \"{ctx['book']}\".",
        f"Working on chapter \"{ctx['chapter']}\" right now. It's told from {ctx['pov']}'s point of view.",
        f"Basically: {ctx['description']}" if ctx['description'] else None,
        f"Should include: {ctx['entities_brief']}" if ctx['entities_brief'] else None,
        ctx['continue_directive'],
    ])),

    # Template 13: Sparse/trusting
    lambda ctx: "\n\n".join(filter(None, [
        f"\"{ctx['book']}\" - {ctx['chapter']}",
        f"POV: {ctx['pov']}. {ctx['description']}" if ctx['description'] else f"POV: {ctx['pov']}",
        ctx['continue_minimal'],
    ])),

    # Template 14: Instructions style
    lambda ctx: "\n\n".join(filter(None, [
        f"Please write the following scene:",
        f"Book: \"{ctx['book']}\"",
        f"Chapter: \"{ctx['chapter']}\"",
        f"Narrator: {ctx['pov']}",
        f"Description: {ctx['description']}" if ctx['description'] else None,
        f"Include:\n{ctx['entities_list']}" if ctx['entities_list'] else None,
        ctx['position_formal'],
        ctx['continue_quote'],
    ])),

    # Template 15: Craft-focused
    lambda ctx: "\n\n".join(filter(None, [
        f"Craft a scene for chapter \"{ctx['chapter']}\" of \"{ctx['book']}\".",
        f"Written from {ctx['pov']}'s point of view, {ctx['entities_prose']}." if ctx['entities_prose'] else f"Written from {ctx['pov']}'s point of view.",
        f"{ctx['description']}" if ctx['description'] else None,
        ctx['position_minimal'],
        ctx['continue_directive'],
    ])),

    # Template 16: Dialogue-like
    lambda ctx: "\n\n".join(filter(None, [
        f"I'm stuck on a scene. Can you write it for me?",
        f"It's chapter \"{ctx['chapter']}\" of my book \"{ctx['book']}\". {ctx['pov']} is the narrator.",
        f"What happens: {ctx['description']}" if ctx['description'] else None,
        ctx['continue_inline'],
    ])),

    # Template 17: Context-heavy
    lambda ctx: "\n\n".join(filter(None, [
        f"Context: \"{ctx['book']}\", chapter \"{ctx['chapter']}\"",
        f"POV character: {ctx['pov']}",
        f"Scene elements: {ctx['entities_prose']}" if ctx['entities_prose'] else None,
        f"What happens: {ctx['description']}" if ctx['description'] else None,
        ctx['position_formal'],
        "Write this scene.",
        ctx['continue_quote'],
    ])),

    # Template 18: Author notes style
    lambda ctx: "\n\n".join(filter(None, [
        f"[Writing \"{ctx['book']}\" - Chapter: \"{ctx['chapter']}\"]",
        f"POV: {ctx['pov']}",
        f"Notes: {ctx['description']}" if ctx['description'] else None,
        f"Include: {ctx['entities_brief']}" if ctx['entities_brief'] else None,
        ctx['position_casual'],
        ctx['continue_minimal'],
    ])),

    # Template 19: Simple continuation
    lambda ctx: "\n\n".join(filter(None, [
        f"Continue writing \"{ctx['book']}\".",
        f"Chapter \"{ctx['chapter']}\", {ctx['pov']}'s perspective.",
        ctx['description'] if ctx['description'] else None,
        ctx['continue_quote'],
    ])),

    # Template 20: Compose style
    lambda ctx: "\n\n".join(filter(None, [
        f"Compose the next scene in my story.",
        f"Novel: \"{ctx['book']}\" | Chapter: \"{ctx['chapter']}\" | Voice: {ctx['pov']}",
        f"Scene summary: {ctx['description']}" if ctx['description'] else None,
        f"Key elements: {ctx['entities_brief']}" if ctx['entities_brief'] else None,
        ctx['position_minimal'],
        ctx['continue_directive'],
    ])),

    # Template 21: Bare bones
    lambda ctx: "\n\n".join(filter(None, [
        f"{ctx['book']} / {ctx['chapter']}",
        f"{ctx['pov']} POV",
        ctx['description'] if ctx['description'] else None,
        ctx['continue_inline'],
    ])),

    # Template 22: Immersive request
    lambda ctx: "\n\n".join(filter(None, [
        f"Step into the world of \"{ctx['book']}\" and write chapter \"{ctx['chapter']}\" through {ctx['pov']}'s eyes.",
        f"{ctx['description']}" if ctx['description'] else None,
        f"Weave in: {ctx['entities_prose']}." if ctx['entities_prose'] else None,
        ctx['continue_quote'],
    ])),

    # Template 23: Task-oriented
    lambda ctx: "\n\n".join(filter(None, [
        f"Task: Write a scene",
        f"Book: {ctx['book']}",
        f"Chapter: {ctx['chapter']}",
        f"POV: {ctx['pov']}",
        f"Plot: {ctx['description']}" if ctx['description'] else None,
        ctx['position_minimal'],
        ctx['continue_minimal'],
    ])),

    # Template 24: Gentle request
    lambda ctx: "\n\n".join(filter(None, [
        f"Would you mind helping me write a scene?",
        f"It's for \"{ctx['book']}\", specifically chapter \"{ctx['chapter']}\". The narrator is {ctx['pov']}.",
        f"Here's what should happen: {ctx['description']}" if ctx['description'] else None,
        ctx['position_casual'],
        ctx['continue_quote'],
    ])),

    # Template 25: No description variant
    lambda ctx: "\n\n".join(filter(None, [
        f"Write chapter \"{ctx['chapter']}\" of \"{ctx['book']}\".",
        f"POV: {ctx['pov']}",
        f"Elements: {ctx['entities_list']}" if ctx['entities_list'] else None,
        ctx['position_formal'],
        ctx['continue_directive'],
    ])),

    # Template 26: Prose-forward
    lambda ctx: "\n\n".join(filter(None, [
        f"I'm writing a novel called \"{ctx['book']}\" and I need help with chapter \"{ctx['chapter']}\", which is narrated by {ctx['pov']}. {ctx['description']}" if ctx['description'] else f"I'm writing a novel called \"{ctx['book']}\" and I need help with chapter \"{ctx['chapter']}\", which is narrated by {ctx['pov']}.",
        f"The scene features {ctx['entities_prose']}." if ctx['entities_prose'] else None,
        ctx['continue_inline'],
    ])),

    # Template 27: Quick and dirty
    lambda ctx: "\n\n".join(filter(None, [
        f"Scene needed: {ctx['chapter']} ({ctx['book']})",
        f"{ctx['pov']} narrates. {ctx['description']}" if ctx['description'] else f"{ctx['pov']} narrates.",
        ctx['continue_minimal'],
    ])),

    # Template 28: Creative partner
    lambda ctx: "\n\n".join(filter(None, [
        f"Let's write together. I'm working on \"{ctx['book']}\".",
        f"Chapter \"{ctx['chapter']}\" - it's from {ctx['pov']}'s point of view.",
        f"The scene: {ctx['description']}" if ctx['description'] else None,
        f"Include these elements if you can: {ctx['entities_brief']}" if ctx['entities_brief'] else None,
        ctx['continue_quote'],
    ])),

    # Template 29: Structured request
    lambda ctx: "\n\n".join(filter(None, [
        f"**Book:** {ctx['book']}",
        f"**Chapter:** {ctx['chapter']}",
        f"**POV:** {ctx['pov']}",
        f"**Summary:** {ctx['description']}" if ctx['description'] else None,
        f"**Elements:** {ctx['entities_brief']}" if ctx['entities_brief'] else None,
        "",
        f"Write this scene. {ctx['position_minimal']}" if ctx['position_minimal'] else "Write this scene.",
        ctx['continue_directive'],
    ])),

    # Template 30: Flow state
    lambda ctx: "\n\n".join(filter(None, [
        f"Keep the story going.",
        f"\"{ctx['book']}\" - chapter \"{ctx['chapter']}\" - {ctx['pov']} narrating.",
        ctx['description'] if ctx['description'] else None,
        ctx['continue_quote'],
    ])),
]


def create_user_prompt(
    frontmatter: dict,
    scene_index: int,
    total_scenes: int,
    previous_scene: Optional[str] = None,
    seed: Optional[str] = None
) -> str:
    """Create the user prompt for a scene using varied templates."""
    book = frontmatter.get('book', 'Unknown Book')
    chapter = frontmatter.get('title', frontmatter.get('chapter', 'Unknown Chapter'))
    description = frontmatter.get('description', '')

    pov = frontmatter.get('pov', [])
    if isinstance(pov, list):
        pov = pov[0] if pov else 'third person'

    # Build context dict with all variations
    ctx = {
        'book': book,
        'chapter': chapter,
        'description': description,
        'pov': pov,
        'entities_list': format_entities(frontmatter, 'list'),
        'entities_prose': format_entities(frontmatter, 'prose'),
        'entities_brief': format_entities(frontmatter, 'brief'),
        'position_formal': get_scene_position_text(scene_index, total_scenes, 'formal'),
        'position_casual': get_scene_position_text(scene_index, total_scenes, 'casual'),
        'position_minimal': get_scene_position_text(scene_index, total_scenes, 'minimal'),
        'continue_quote': get_continuation_text(previous_scene, 'quote'),
        'continue_inline': get_continuation_text(previous_scene, 'inline'),
        'continue_directive': get_continuation_text(previous_scene, 'directive'),
        'continue_minimal': get_continuation_text(previous_scene, 'minimal'),
    }

    # Deterministic template selection based on content hash
    if seed:
        hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    else:
        hash_val = int(hashlib.md5(f"{book}{chapter}{scene_index}".encode()).hexdigest(), 16)

    template_idx = hash_val % len(PROMPT_TEMPLATES)
    template = PROMPT_TEMPLATES[template_idx]

    return template(ctx)


def chapter_to_chatml(file_path: Path) -> list[dict]:
    """Convert a chapter file to ChatML training examples."""
    content = file_path.read_text(encoding='utf-8')
    frontmatter, remaining = parse_frontmatter(content)

    if not frontmatter:
        print(f"Warning: No frontmatter in {file_path}", file=sys.stderr)
        return []

    # Skip files without required fields
    if not frontmatter.get('book') and not frontmatter.get('series'):
        print(f"Warning: Missing book/series in {file_path}", file=sys.stderr)
        return []
    if not frontmatter.get('description'):
        print(f"Warning: Missing description in {file_path}", file=sys.stderr)
        return []

    prose = clean_prose(remaining)
    if not prose:
        print(f"Warning: No prose in {file_path}", file=sys.stderr)
        return []

    scenes = split_into_scenes(prose)
    if not scenes:
        scenes = [prose]

    book = frontmatter.get('book', '')
    chapter = frontmatter.get('title', frontmatter.get('chapter', ''))

    examples = []
    for i, scene in enumerate(scenes):
        previous_scene = scenes[i - 1] if i > 0 else None

        # Create seed for deterministic template selection
        seed = f"{book}{chapter}{i}"
        hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        template_idx = hash_val % len(PROMPT_TEMPLATES)

        user_prompt = create_user_prompt(
            frontmatter=frontmatter,
            scene_index=i,
            total_scenes=len(scenes),
            previous_scene=previous_scene,
            seed=seed
        )

        example = {
            "conversations": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": scene}
            ],
            "metadata": {
                "source_file": str(file_path.name),
                "book": book,
                "chapter": frontmatter.get('chapter', ''),
                "scene": i + 1,
                "total_scenes": len(scenes),
                "template": template_idx + 1  # 1-indexed for readability
            }
        }
        examples.append(example)

    return examples


def process_directory(dir_path: Path) -> list[dict]:
    """Process all markdown files in a directory recursively."""
    all_examples = []
    md_files = sorted(dir_path.rglob('*.md'))

    for md_file in md_files:
        # Skip non-chapter files
        if md_file.name.startswith('🗺') or md_file.name.startswith('.'):
            continue
        if any(skip in str(md_file) for skip in ['/Archive/', '\\Archive\\', '/Scenes/', '\\Scenes\\', '/The World/', '\\The World\\']):
            continue

        print(f"Processing: {md_file.name}", file=sys.stderr)
        examples = chapter_to_chatml(md_file)
        all_examples.extend(examples)

    return all_examples


def main():
    parser = argparse.ArgumentParser(description='Convert Obsidian vault chapters to ChatML')
    parser.add_argument('input', type=Path, help='Input file or directory')
    parser.add_argument('--output', '-o', type=Path, help='Output JSONL file')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process directory recursively')
    parser.add_argument('--preview', '-p', action='store_true', help='Preview without writing')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.input.is_dir():
        if not args.recursive:
            print("Error: Use --recursive for directories", file=sys.stderr)
            sys.exit(1)
        examples = process_directory(args.input)
    else:
        examples = chapter_to_chatml(args.input)

    if not examples:
        print("No training examples generated.", file=sys.stderr)
        sys.exit(1)

    print(f"\nGenerated {len(examples)} training examples", file=sys.stderr)

    if args.preview:
        print("\n" + "=" * 60)
        print("PREVIEW - First Example:")
        print("=" * 60)
        print(json.dumps(examples[0], indent=2))
        if len(examples) > 1:
            print(f"\n... and {len(examples) - 1} more examples")
    elif args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"Wrote {len(examples)} examples to {args.output}", file=sys.stderr)
    else:
        for example in examples:
            print(json.dumps(example, ensure_ascii=False))


if __name__ == '__main__':
    main()
