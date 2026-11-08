"""Unchanged native answer parsing, judging and scoring."""
from .native import harness as upstream, metrics

def response_text(result):
    choice = result["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Incomplete reader/judge response")
    text = choice["message"].get("content")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty reader/judge response")
    return text


def grade(question, answer, client):
    parsed = metrics.extract_boxed_answer(answer)
    name = metrics.eval_name(question["eval_function"])
    judge_messages = None
    judge_output = None
    if name in upstream.LLM_EVAL_FUNCTIONS:
        builder = (
            metrics._build_abstention_judge_messages
            if name == "llm_abstention_checker"
            else metrics._build_gotchas_judge_messages
        )
        judge_messages = builder(
            question_text=question["question"],
            reference_answer=question["answer"],
            model_full_response=answer,
            model_final_answer=parsed,
        )
        judge_output = response_text(
            client.post(
                "v1/chat/completions",
                {
                    "model": "openai/gpt-5.2",
                    "messages": judge_messages,
                    "max_tokens": 4096,
                    "reasoning": {"effort": "medium"},
                    "provider": {"require_parameters": True},
                },
            )
        )
        label, reason = metrics._parse_llm_binary_judgement(judge_output)
        score = bool(label)
    else:
        score = metrics.score_to_bool(
            metrics.eval_from_spec(
                question["eval_function"], parsed, question["answer"]
            )
        )
        reason = None
    if metrics.is_unknown(parsed):
        score = False
    return {
        "score": int(score),
        "parsed_answer": parsed,
        "judge_messages": judge_messages,
        "judge_output": judge_output,
        "judge_reason": reason,
        "eval_function": question["eval_function"],
    }
