"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.helpers import validate_node_labels


def _validate_entity_labels(labels: str | list[str]) -> list[str]:
    resolved_labels = labels.split(':') if isinstance(labels, str) else labels
    filtered_labels = [label for label in resolved_labels if label]
    validate_node_labels(filtered_labels)
    return filtered_labels


def get_episode_node_save_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MERGE (n:Episodic {uuid: $uuid})
            SET
                n.name = $name,
                n.group_id = $group_id,
                n.created_at = $created_at,
                n.source = $source,
                n.source_description = $source_description,
                n.content = $content,
                n.valid_at = $valid_at,
                n.entity_edges = $entity_edges
            RETURN n.uuid AS uuid
        """

    return """
        MERGE (n:Episodic {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content,
        entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
        RETURN n.uuid AS uuid
    """


def get_episode_node_save_bulk_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MERGE (n:Episodic {uuid: $uuid})
            SET
                n.name = $name,
                n.group_id = $group_id,
                n.created_at = $created_at,
                n.source = $source,
                n.source_description = $source_description,
                n.content = $content,
                n.valid_at = $valid_at,
                n.entity_edges = $entity_edges
            RETURN n.uuid AS uuid
        """

    return """
        UNWIND $episodes AS episode
        MERGE (n:Episodic {uuid: episode.uuid})
        SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, source: episode.source, content: episode.content,
        entity_edges: episode.entity_edges, created_at: episode.created_at, valid_at: episode.valid_at}
        RETURN n.uuid AS uuid
    """


EPISODIC_NODE_RETURN = """
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.created_at AS created_at,
    e.source AS source,
    e.source_description AS source_description,
    e.content AS content,
    e.valid_at AS valid_at,
    e.entity_edges AS entity_edges
"""


def get_entity_node_save_query(provider: GraphProvider, labels: str) -> str:
    validated_labels = _validate_entity_labels(labels)
    labels = ':'.join(validated_labels)

    if provider == GraphProvider.KUZU:
        return """
            MERGE (n:Entity {uuid: $uuid})
            SET
                n.name = $name,
                n.group_id = $group_id,
                n.labels = $labels,
                n.created_at = $created_at,
                n.name_embedding = $name_embedding,
                n.summary = $summary,
                n.attributes = $attributes
            RETURN n.uuid AS uuid
        """

    return (
        f"""
        MERGE (n:Entity {{uuid: $entity_data.uuid}})
        SET n:{labels}
        SET n = $entity_data
        """
        + 'WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $entity_data.name_embedding)'
        + """
        RETURN n.uuid AS uuid
    """
    )


def get_entity_node_save_bulk_query(provider: GraphProvider, nodes: list[dict]) -> str | Any:
    for node in nodes:
        _validate_entity_labels(node.get('labels', []))

    if provider == GraphProvider.KUZU:
        return """
            MERGE (n:Entity {uuid: $uuid})
            SET
                n.name = $name,
                n.group_id = $group_id,
                n.labels = $labels,
                n.created_at = $created_at,
                n.name_embedding = $name_embedding,
                n.summary = $summary,
                n.attributes = $attributes
            RETURN n.uuid AS uuid
        """

    return (
        """
            UNWIND $nodes AS node
            MERGE (n:Entity {uuid: node.uuid})
            SET n:$(node.labels)
            SET n = node
            """
        + 'WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)'
        + """
        RETURN n.uuid AS uuid
    """
    )


def get_entity_node_return_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            n.uuid AS uuid,
            n.name AS name,
            n.group_id AS group_id,
            n.labels AS labels,
            n.created_at AS created_at,
            n.summary AS summary,
            n.attributes AS attributes
        """

    return """
        n.uuid AS uuid,
        n.name AS name,
        n.group_id AS group_id,
        n.created_at AS created_at,
        n.summary AS summary,
        labels(n) AS labels,
        properties(n) AS attributes
    """


def get_community_node_save_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MERGE (n:Community {uuid: $uuid})
            SET
                n.name = $name,
                n.group_id = $group_id,
                n.created_at = $created_at,
                n.name_embedding = $name_embedding,
                n.summary = $summary
            RETURN n.uuid AS uuid
        """

    return """
        MERGE (n:Community {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at}
        WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $name_embedding)
        RETURN n.uuid AS uuid
    """


COMMUNITY_NODE_RETURN = """
    c.uuid AS uuid,
    c.name AS name,
    c.group_id AS group_id,
    c.created_at AS created_at,
    c.name_embedding AS name_embedding,
    c.summary AS summary
"""


def get_saga_node_save_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MERGE (n:Saga {uuid: $uuid})
            SET
                n.name = $name,
                n.group_id = $group_id,
                n.created_at = $created_at,
                n.summary = $summary,
                n.first_episode_uuid = $first_episode_uuid,
                n.last_episode_uuid = $last_episode_uuid,
                n.last_summarized_at = $last_summarized_at
            RETURN n.uuid AS uuid
        """

    return """
        MERGE (n:Saga {uuid: $uuid})
        SET n = {uuid: $uuid, name: $name, group_id: $group_id, created_at: $created_at, summary: $summary, first_episode_uuid: $first_episode_uuid, last_episode_uuid: $last_episode_uuid, last_summarized_at: $last_summarized_at}
        RETURN n.uuid AS uuid
    """


SAGA_NODE_RETURN = """
    s.uuid AS uuid,
    s.name AS name,
    s.group_id AS group_id,
    s.created_at AS created_at,
    s.summary AS summary,
    s.first_episode_uuid AS first_episode_uuid,
    s.last_episode_uuid AS last_episode_uuid,
    s.last_summarized_at AS last_summarized_at
"""
