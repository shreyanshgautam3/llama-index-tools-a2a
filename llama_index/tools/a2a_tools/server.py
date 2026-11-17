"""
Inbound A2A Server - exposes a LlamaIndex Workflow as an A2A agent.

Mirrors the `workflow_as_mcp` pattern from llama-index-tools-mcp.
Uses the a2a-sdk server components (AgentExecutor, DefaultRequestHandler, etc.)
to create a FastAPI app that conforms to the A2A protocol.

Supports both A2A protocol v0.3 and v1.0 via method name handling.
"""

from typing import Any

from llama_index.core.workflow import StartEvent, Workflow
from pydantic import BaseModel

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities as A2AAgentCapabilities,
)
from a2a.types import (
    AgentCard as A2AAgentCard,
)
from a2a.types import (
    AgentProvider as A2AAgentProvider,
)
from a2a.types import (
    AgentSkill,
    Artifact,
    Message,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

# Use our local AgentCard model which has supported_interfaces for v1.0 wire format
from .models import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    Skill,
)


class WorkflowAgentExecutor(AgentExecutor):
    """
    AgentExecutor that wraps a LlamaIndex Workflow.

    The workflow's StartEvent fields become the input schema,
    and the workflow execution produces the output streamed via events.
    """

    def __init__(
        self,
        workflow: Workflow,
        start_event_model: type[BaseModel] | None = None,
        workflow_name: str | None = None,
        workflow_description: str | None = None,
    ):
        self.workflow = workflow
        self.start_event_model = start_event_model or workflow._start_event_class
        self.workflow_name = workflow_name or workflow.__class__.__name__
        self.workflow_description = (
            workflow_description or workflow.__doc__ or "LlamaIndex Workflow"
        )

        if self.start_event_model == StartEvent:
            raise ValueError(
                "Must declare a custom StartEvent class in your workflow "
                "or provide a start_event_model."
            )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the workflow, streaming intermediate and final events."""
        user_input = context.get_user_input()
        if not user_input:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(state=TaskState.failed),
                    final=True,
                )
            )
            return

        try:
            # Parse input into the workflow's StartEvent model
            import json

            if isinstance(user_input, str):
                try:
                    input_data = json.loads(user_input)
                except json.JSONDecodeError as e:
                    # Single-argument workflow
                    fields = self.start_event_model.model_fields
                    if len(fields) == 1:
                        field_name = list(fields.keys())[0]
                        input_data = {field_name: user_input}
                    else:
                        raise ValueError(f"Expected JSON input, got: {user_input}") from e
            elif isinstance(user_input, dict):
                input_data = user_input
            else:
                input_data = {"input": user_input}

            # Validate and create StartEvent
            start_event = self.start_event_model.model_validate(input_data)

            # Run the workflow (Workflow.run() returns a coroutine — await it directly)
            result = await self.workflow.run(start_event=start_event)

            # Send artifact with result
            if result is not None:
                try:
                    result_text = result.model_dump_json()
                except AttributeError:
                    result_text = str(result)

                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        task_id=context.task_id,
                        context_id=context.context_id,
                        artifact=Artifact(
                            artifact_id=f"{context.task_id}-artifact",
                            parts=[TextPart(text=result_text)],
                        ),
                        last_chunk=True,
                    )
                )

            # Final completion status
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(state=TaskState.completed),
                    final=True,
                )
            )

        except Exception as e:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=Message(
                            messageId=f"{context.task_id}-error",
                            role=Role.agent,
                            parts=[TextPart(text=f"Workflow error: {str(e)}")],
                        ),
                    ),
                    final=True,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the workflow execution."""
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )


def create_agent_card(
    workflow: Workflow,
    workflow_name: str | None = None,
    workflow_description: str | None = None,
    start_event_model: type[BaseModel] | None = None,
    url: str = "http://localhost:8000",
    version: str = "1.0.0",
    protocol_version: str = "0.3",
) -> AgentCard:
    """Create an AgentCard from a LlamaIndex Workflow.

    Args:
        protocol_version: "0.3" for v0.3 format (url + preferred_transport).
                          "1.0" is NOT SUPPORTED by a2a-sdk 0.3.26.
                          The underlying a2a-sdk 0.3.26 only handles v0.3 method names
                          (message/send, tasks/get). Do not use "1.0" - it will raise NotImplementedError.
    """
    if protocol_version == "1.0":
        raise NotImplementedError(
            "protocol_version='1.0' is not supported by a2a-sdk 0.3.26. "
            "The server can only handle v0.3 method names (message/send, tasks/get). "
            "Use protocol_version='0.3' (default)."
        )
    name = workflow_name or workflow.__class__.__name__
    description = workflow_description or workflow.__doc__ or "LlamaIndex Workflow Agent"

    start_event_cls = start_event_model or workflow._start_event_class

    if start_event_cls != StartEvent:
        start_event_cls.model_json_schema()
    else:
        pass

    skill = AgentSkill(
        id=name.lower().replace(" ", "-"),
        name=name,
        description=description,
        tags=["workflow", "llama-index"],
        examples=[f"Run {name} workflow"],
        input_modes=["text/plain", "application/json"],
        output_modes=["text/plain", "application/json"],
    )

    # Create our local AgentCard model for discovery/serialization
    capabilities = AgentCapabilities(streaming=False, push_notifications=False)
    provider = AgentProvider(
        organization="LlamaIndex", url="https://github.com/run-llama/llama_index"
    )

    # v0.3 format only (v1.0 not supported by a2a-sdk 0.3.26)
    agent_card = AgentCard(
        name=name,
        description=description,
        version=version,
        url=url,
        skills=[Skill(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            tags=skill.tags,
            examples=skill.examples,
            input_modes=skill.input_modes,
            output_modes=skill.output_modes,
            # input_schema and output_schema are not standard A2A fields; omit them
        )],
        capabilities=capabilities,
        provider=provider,
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        preferred_transport="JSONRPC",
    )

    return agent_card


def _to_a2a_agent_card(local_card: AgentCard, protocol_version: str = "0.3") -> A2AAgentCard:
    """Convert our local AgentCard to a2a-sdk AgentCard for A2AFastAPIApplication."""
    # Convert skills
    a2a_skills = []
    for s in local_card.skills:
        a2a_skills.append(AgentSkill(
            id=s.id,
            name=s.name,
            description=s.description,
            tags=s.tags or ["workflow", "llama-index"],
            examples=[],
            input_modes=s.input_modes or ["text/plain", "application/json"],
            output_modes=s.output_modes or ["text/plain", "application/json"],
        ))

    # Convert capabilities
    a2a_capabilities = A2AAgentCapabilities(
        streaming=local_card.capabilities.streaming if local_card.capabilities else False,
        push_notifications=(
            local_card.capabilities.push_notifications if local_card.capabilities else False
        ),
    )

    # Convert provider
    a2a_provider = A2AAgentProvider(
        organization=local_card.provider.organization if local_card.provider else "LlamaIndex",
        url=local_card.provider.url if local_card.provider else "https://github.com/run-llama/llama_index",
    )

    # Build a2a-sdk AgentCard (v0.3 format only)
    a2a_card_data = {
        "name": local_card.name,
        "description": local_card.description,
        "version": local_card.version,
        "url": local_card.url,
        "skills": a2a_skills,
        "capabilities": a2a_capabilities,
        "provider": a2a_provider,
        "default_input_modes": local_card.default_input_modes or ["text/plain", "application/json"],
        "default_output_modes": local_card.default_output_modes or ["text/plain", "application/json"],
        "preferred_transport": local_card.preferred_transport or "JSONRPC",
    }

    return A2AAgentCard(**a2a_card_data)


def workflow_as_a2a(
    workflow: Workflow,
    workflow_name: str | None = None,
    workflow_description: str | None = None,
    start_event_model: type[BaseModel] | None = None,
    url: str = "http://localhost:8000",
    version: str = "1.0.0",
    **app_kwargs: Any,
) -> Any:
    """
    Convert a LlamaIndex Workflow to an A2A-compatible FastAPI application.

    Mirrors the `workflow_as_mcp` pattern from llama-index-tools-mcp.

    Args:
        workflow: The LlamaIndex Workflow to expose as an A2A agent.
        workflow_name: Name for the agent (defaults to workflow class name).
        workflow_description: Description (defaults to workflow docstring).
        start_event_model: The StartEvent model for the workflow.
                           Defaults to workflow._start_event_class.
        url: Base URL where the agent will be hosted.
        version: Version string for the agent card.
        **app_kwargs: Additional kwargs passed to FastAPI constructor.

    Returns:
        A FastAPI application with A2A routes configured.

    Note:
        The underlying a2a-sdk 0.3.26 only supports v0.3 method names
        (message/send, tasks/get). v1.0 AgentCard format is not supported.

    Example:
        >>> from llama_index.core.workflow import Workflow, StartEvent, StopEvent
        >>> from llama_index.tools.a2a import workflow_as_a2a
        >>>
        >>> class MyStartEvent(StartEvent):
        ...     query: str
        >>>
        >>> class MyWorkflow(Workflow):
        ...     async def run(self, start_event: MyStartEvent) -> StopEvent:
        ...         return StopEvent(result=f"Processed: {start_event.query}")
        >>>
        >>> app = workflow_as_a2a(MyWorkflow())
        >>> # Run: uvicorn my_app:app --host 0.0.0.0 --port 8000
    """
    # Create AgentCard (local model for discovery/serialization) - v0.3 only
    local_agent_card = create_agent_card(
        workflow=workflow,
        workflow_name=workflow_name,
        workflow_description=workflow_description,
        start_event_model=start_event_model,
        url=url,
        version=version,
        protocol_version="0.3",
    )

    # Convert to a2a-sdk AgentCard for A2AFastAPIApplication
    a2a_agent_card = _to_a2a_agent_card(local_agent_card, "0.3")

    # Create executor
    executor = WorkflowAgentExecutor(
        workflow=workflow,
        start_event_model=start_event_model,
        workflow_name=workflow_name,
        workflow_description=workflow_description,
    )

    # Create task store
    task_store = InMemoryTaskStore()

    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    # Build the A2A FastAPI application
    app = A2AFastAPIApplication(
        agent_card=a2a_agent_card,
        http_handler=request_handler,
    ).build(**app_kwargs)

    # Store references on app state for programmatic access (use local card for consistency)
    app.state.agent_card = local_agent_card
    app.state.executor = executor
    app.state.task_store = task_store
    app.state.request_handler = request_handler

    # Override agent-card route to serve local card (with supportedInterfaces for v1.0)
    async def _serve_local_agent_card():
        # Serialize local card using by_alias=True for camelCase wire format
        return local_agent_card.model_dump(by_alias=True, exclude_none=True)

    # Find and replace the agent-card route
    for i, route in enumerate(app.routes):
        if hasattr(route, 'path') and route.path == '/.well-known/agent-card.json':
            app.routes[i] = type(route)(
                path=route.path,
                endpoint=_serve_local_agent_card,
                methods=route.methods,
                name=route.name,
            )
            break

    return app
