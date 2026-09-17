# AdoptAI dashboard design system

## Product character

AdoptAI is a calm, technical monitoring product—not an alarm console. The interface should feel precise, trustworthy, and lightweight. Dark mode uses a navy shell and ink-blue panels; light mode uses an off-white canvas and crisp white surfaces. Both retain the cyan identity and restrained semantic colors. The product name is always paired with the subtitle “Intelligent Computer Performance Monitor.”

## Visual tokens

- Dark: background `#07111F`; surface `#0D1A2B`; secondary surface `#112238`; primary text `#F4F8FC`; secondary text `#9EB0C5`.
- Light: background `#F7F9FC`; surface `#FFFFFF`; secondary surface `#F1F5F9`; primary text `#142033`; secondary text `#526176`.
- Brand/accent: `#35C6D0`; accent-soft: `rgba(53, 198, 208, .12)`.
- Success: `#2FA875`; warning: `#D99827`; danger: `#D95361`; semantic meaning stays identical in both themes.
- Borders and chart grids are low-contrast theme tokens; shadows are softer in light mode.
- Cards use a 16–18 px radius. Controls use a 10–12 px radius.
- Typography uses Inter where available, then system sans-serif. Numeric scores use tabular numerals.

## Layout and hierarchy

1. A compact product masthead establishes the product and appearance control. It always clears Streamlit's fixed toolbar with at least 4rem of top space.
2. Live Monitor gives the Risk Score the strongest visual weight. Current metrics are secondary cards.
3. Use a maximum content width near 1,400 px with generous vertical rhythm.
4. Three tabs only: Live Monitor, History, Model Info.
5. Charts cover 5–10 minutes and use consistent colors: CPU cyan, RAM violet, swap amber, disk latency coral, risk score cyan/red.

## Interaction rules

- Start is primary only when stopped. When running, Start becomes disabled/muted and Stop becomes the enabled primary action.
- Theme preference is kept in Streamlit session state and mirrored in the URL query parameter. Switching it never touches collector or inference state.
- The recording strip uses the active run's SQLite row count. When stopped it shows the most recent run count.
- Never show a score before 120 seconds of continuous segment history. Use a progress bar and plain-language warm-up state.
- “Risk Score” is always described as a model score out of 100, never as a probability.
- The binary boundary stays fixed at 50. Presentation bands do not alter prediction logic.
- Alerts are visible but do not flash. Duplicate alerts are suppressed until a state transition or five-minute cooldown.
- Errors show a short user-facing message. Technical details go to the application/collector log.

## Content rules

- Prefer sentence case and compact labels.
- Show units with every metric.
- Truncate UUIDs visually while preserving the full value in captions or tables.
- Model Info must say “Experimental prototype” and state the distribution-shift limitation.
- Avoid unexplained ML vocabulary on Live Monitor. Technical details belong on Model Info.

## Accessibility and responsive behavior

- Maintain readable contrast on both dark and light surfaces.
- Never rely on color alone: every state includes a text label and icon/dot.
- Use responsive Streamlit columns that stack naturally on narrow screens.
- Keep touch targets at least 40 px high and avoid dense control clusters.
