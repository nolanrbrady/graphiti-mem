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

from graphiti_core.driver.driver import GraphProvider

EPISODIC_EDGE_SAVE = """
    MATCH (episode:Episodic {uuid: $episode_uuid})
    MATCH (node:Entity {uuid: $entity_uuid})
    MERGE (episode)-[e:MENTIONS {uuid: $uuid}]->(node)
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""


def get_episodic_edge_save_bulk_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MATCH (episode:Episodic {uuid: $source_node_uuid})
            MATCH (node:Entity {uuid: $target_node_uuid})
            MERGE (episode)-[e:MENTIONS {uuid: $uuid}]->(node)
            SET
                e.group_id = $group_id,
                e.created_at = $created_at
            RETURN e.uuid AS uuid
        """

    return """
        UNWIND $episodic_edges AS edge
        MATCH (episode:Episodic {uuid: edge.source_node_uuid})
        MATCH (node:Entity {uuid: edge.target_node_uuid})
        MERGE (episode)-[e:MENTIONS {uuid: edge.uuid}]->(node)
        SET
            e.group_id = edge.group_id,
            e.created_at = edge.created_at
        RETURN e.uuid AS uuid
    """


EPISODIC_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


def get_entity_edge_save_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MATCH (source:Entity {uuid: $source_uuid})
            MATCH (target:Entity {uuid: $target_uuid})
            MERGE (source)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(target)
            SET
                e.group_id = $group_id,
                e.created_at = $created_at,
                e.name = $name,
                e.fact = $fact,
                e.fact_embedding = $fact_embedding,
                e.episodes = $episodes,
                e.expired_at = $expired_at,
                e.valid_at = $valid_at,
                e.invalid_at = $invalid_at,
                e.reference_time = $reference_time,
                e.attributes = $attributes
            RETURN e.uuid AS uuid
        """

    return (
        """
            MATCH (source:Entity {uuid: $edge_data.source_uuid})
            MATCH (target:Entity {uuid: $edge_data.target_uuid})
            MERGE (source)-[e:RELATES_TO {uuid: $edge_data.uuid}]->(target)
            SET e = $edge_data
            """
        + """WITH e CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", $edge_data.fact_embedding)"""
        + """
        RETURN e.uuid AS uuid
        """
    )


def get_entity_edge_save_bulk_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MATCH (source:Entity {uuid: $source_node_uuid})
            MATCH (target:Entity {uuid: $target_node_uuid})
            MERGE (source)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(target)
            SET
                e.group_id = $group_id,
                e.created_at = $created_at,
                e.name = $name,
                e.fact = $fact,
                e.fact_embedding = $fact_embedding,
                e.episodes = $episodes,
                e.expired_at = $expired_at,
                e.valid_at = $valid_at,
                e.invalid_at = $invalid_at,
                e.reference_time = $reference_time,
                e.attributes = $attributes
            RETURN e.uuid AS uuid
        """

    return (
        """
            UNWIND $entity_edges AS edge
            MATCH (source:Entity {uuid: edge.source_node_uuid})
            MATCH (target:Entity {uuid: edge.target_node_uuid})
            MERGE (source)-[e:RELATES_TO {uuid: edge.uuid}]->(target)
            SET e = edge
            """
        + 'WITH e, edge CALL db.create.setRelationshipVectorProperty(e, "fact_embedding", edge.fact_embedding)'
        + """
            RETURN edge.uuid AS uuid
        """
    )


def get_entity_edge_return_query(provider: GraphProvider) -> str:
    return """
        e.uuid AS uuid,
        n.uuid AS source_node_uuid,
        m.uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
    """ + (
        'e.attributes AS attributes'
        if provider == GraphProvider.KUZU
        else 'properties(e) AS attributes'
    )


def get_community_edge_save_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MATCH (community:Community {uuid: $community_uuid})
            MATCH (node:Entity {uuid: $entity_uuid})
            MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
            SET
                e.group_id = $group_id,
                e.created_at = $created_at
            RETURN e.uuid AS uuid
            UNION
            MATCH (community:Community {uuid: $community_uuid})
            MATCH (node:Community {uuid: $entity_uuid})
            MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
            SET
                e.group_id = $group_id,
                e.created_at = $created_at
            RETURN e.uuid AS uuid
        """

    return """
        MATCH (community:Community {uuid: $community_uuid})
        MATCH (node:Entity | Community {uuid: $entity_uuid})
        MERGE (community)-[e:HAS_MEMBER {uuid: $uuid}]->(node)
        SET e = {uuid: $uuid, group_id: $group_id, created_at: $created_at}
        RETURN e.uuid AS uuid
    """


COMMUNITY_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


HAS_EPISODE_EDGE_SAVE = """
    MATCH (saga:Saga {uuid: $saga_uuid})
    MATCH (episode:Episodic {uuid: $episode_uuid})
    MERGE (saga)-[e:HAS_EPISODE {uuid: $uuid}]->(episode)
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""

HAS_EPISODE_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""


NEXT_EPISODE_EDGE_SAVE = """
    MATCH (source_episode:Episodic {uuid: $source_episode_uuid})
    MATCH (target_episode:Episodic {uuid: $target_episode_uuid})
    MERGE (source_episode)-[e:NEXT_EPISODE {uuid: $uuid}]->(target_episode)
    SET
        e.group_id = $group_id,
        e.created_at = $created_at
    RETURN e.uuid AS uuid
"""

NEXT_EPISODE_EDGE_RETURN = """
    e.uuid AS uuid,
    e.group_id AS group_id,
    n.uuid AS source_node_uuid,
    m.uuid AS target_node_uuid,
    e.created_at AS created_at
"""
