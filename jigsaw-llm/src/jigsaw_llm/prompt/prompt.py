import jinja2
import pandas as pd

prompt_ver1: jinja2.Template = jinja2.Template("""# Task
- Determine whether the comment shown to the target violates any specific rules of the subreddit
- If it violates the rules, output 'Yes'. If it does not violate the rules, output 'No'.

## SubReddit
r/{{ subreddit }}

## Rule
{{ rule }}

## Examples of comments that violate the rule
{{ positive_examples }}

## Examples of comments that do not violate the rule
{{ negative_examples }}

# Target Comment
{{ body }}

# Output
""")


class PromptVer1:
    def to_prompt(self, row: pd.Series) -> str:
        return prompt_ver1.render(
            subreddit=row["subreddit"],
            rule=row["rule"],
            positive_examples=row["positive_examples"],
            negative_examples=row["negative_examples"],
            body=row["body"],
        )


prompt_ver2: jinja2.Template = jinja2.Template("""# Task
- Determine whether the comment shown to the target violates any specific rules of the subreddit
- If it violates the rules, output 'Yes'. If it does not violate the rules, output 'No'.
- Example may not exist.

## SubReddit
r/{{ subreddit }}

## Rule
{{ rule }}

## Examples of comments that violate the rule
{{ positive_examples }}

## Examples of comments that do not violate the rule
{{ negative_examples }}

# Target Comment
{{ body }}

# Output
""")


class PromptVer2:
    def to_prompt(self, row: pd.Series) -> str:
        return prompt_ver2.render(
            subreddit=row["subreddit"],
            rule=row["rule"],
            positive_examples=row["positive_examples"],
            negative_examples=row["negative_examples"],
            body=row["body"],
        )


prompt_ver3: jinja2.Template = jinja2.Template("""# Task
- Determine whether the comment shown to the target violates any specific rules of the subreddit
- If it violates the rules, output 'Yes'. If it does not violate the rules, output 'No'.

## SubReddit
r/{{ subreddit }}

## Rule
{{ rule }}

# Target Comment
{{ body }}

# Output
""")


class PromptVer3:
    def to_prompt(self, row: pd.Series) -> str:
        return prompt_ver3.render(
            subreddit=row["subreddit"],
            rule=row["rule"],
            body=row["body"],
        )


prompt_ver4: jinja2.Template = jinja2.Template("""# Task
- Determine whether the comment shown to the target violates any specific rules
- If it violates the rules, output 'Yes'. If it does not violate the rules, output 'No'.

# Rule
{{ rule }}

# Target Comment
{{ body }}

# Output
""")


class PromptVer4:
    def to_prompt(self, row: pd.Series) -> str:
        return prompt_ver4.render(
            rule=row["rule"],
            body=row["body"],
        )


prompt_ver5: jinja2.Template = jinja2.Template("""# Task
- Determine whether the comment shown to the target violates any specific rules
- If it violates the rules, output 'Yes'. If it does not violate the rules, output 'No'.

# Rule
{{ rule }}

# Example
## Violate
{{ positive_example }}

## Not Violate
{{ negative_example }}

# Target Comment
{{ body }}

# Output
""")


class PromptVer5:
    def to_prompt(self, row: pd.Series) -> str:
        if row["positive_example"] != "" or row["negative_example"] != "":
            return prompt_ver5.render(
                rule=row["rule"],
                positive_example=row["positive_example"],
                negative_example=row["negative_example"],
                body=row["body"],
            )

        return prompt_ver4.render(
            rule=row["rule"],
            body=row["body"],
        )


prompt_ver6: jinja2.Template = jinja2.Template("""# Task
- Determine whether the comment shown to the target violates any specific rules
- If it violates the rules, output 'Yes'. If it does not violate the rules, output 'No'.

# Rule
{{ rule }}

# Target Comment
subreddit: r/{{ subreddit }}
{{ body }}

# Output
""")


class PromptVer6:
    def to_prompt(self, row: pd.Series) -> str:
        return prompt_ver4.render(
            rule=row["rule"],
            body=row["body"],
            subreddit=row["subreddit"],
        )


prompt_ver7: jinja2.Template = jinja2.Template("""# Task
- Determine whether the comment shown to the target violates any specific rules
- If it violates the rules, output 'Yes'. If it does not violate the rules, output 'No'.

# Rule
{{ rule }}
{{ rule_context }}

# Target Comment
{{ body }}

# Output
""")


class PromptVer7:
    def to_prompt(self, row: pd.Series) -> str:
        return prompt_ver7.render(
            rule=row["rule"],
            body=row["body"],
            rule_context=row["rule_context"],
        )


# for shieldgemma-9b
prompt_ver8: jinja2.Template = jinja2.Template("""You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {{ body }}
<end_of_turn>

Our safety principle is defined in the below:

* {{ rule }}

===

Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'. And then walk through step by step to be sure we answer correctly.""")  # noqa


class PromptVer8:
    def to_prompt(self, row: pd.Series) -> str:
        return prompt_ver8.render(
            rule=row["rule"],
            body=row["body"],
        )


# for Qwen3Guard
prompt_ver9: jinja2.Template = jinja2.Template("""Task:
1) Determine whether the **USER's query** shown to the target violates any specific rules

<BEGIN SAFETY POLICY>
Yes: Violates the rule.
No: Not Violate the rule.
<END SAFETY POLICY>

<BEGIN RULE>
{{ rule }}
<END RULE>

<BEGIN CONVERSATION>

USER: {{ body }}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST **USER's query** in the above conversation:
- The first line must be one of: 'Yes' or 'No'
""")

class PromptVer9:
    def to_prompt(self, row: pd.Series) -> str:
        return prompt_ver9.render(
            rule=row["rule"],
            body=row["body"],
        )