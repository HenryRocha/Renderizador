#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Henry Rocha e Rafael dos Santos.
Disciplina: Computação Gráfica
Data:
"""

import time
from typing import Dict, List, Tuple

from PIL.Image import NORMAL  # Para operações com tempo

import gpu  # Simula os recursos de uma GPU
import numpy as np


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01  # plano de corte próximo
    far = 1000  # plano de corte distante

    world_transform_stack = []  # pilha de transformações do mundo

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.screen_mat = np.array(
            [
                [width * 2 / 2, 0, 0, width * 2 / 2],
                [0, -height * 2 / 2, 0, height * 2 / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        GL.framebuffer = np.zeros([width * 2, height * 2, 3], dtype=np.uint)

        GL.has_light = False
        GL.light_ambient_intensity = np.zeros([1, 3])
        GL.light_dir = np.zeros([1, 3])
        GL.light_intensity = 0
        GL.light_color = np.zeros([1, 3])
        GL.location = np.zeros([1, 3])
        GL.view_vector = np.array([0, 0, 1])

    @staticmethod
    def create_world_triangles(points: List[float]) -> Tuple[Tuple[Tuple[float]]]:
        """
        Create a tuple where each element is another tuple with 3 tuples.

        (
            (
                (a_x, a_y, a_z),
                (b_x, b_y, b_z),
                (c_x, c_y, c_z)
            ),
            (
                (a_x, a_y, a_z),
                (b_x, b_y, b_z),
                (c_x, c_y, c_z)
            ),
            ...
        )
        """

        triangles = []
        for i in range(0, len(points), 9):
            p_a = np.array([[points[i]], [points[i + 1]], [points[i + 2]], [1]])
            p_b = np.array([[points[i + 3]], [points[i + 4]], [points[i + 5]], [1]])
            p_c = np.array([[points[i + 6]], [points[i + 7]], [points[i + 8]], [1]])

            triangles.append(
                (
                    tuple(GL.mat_mundo.dot(p_a).flatten()),
                    tuple(GL.mat_mundo.dot(p_b).flatten()),
                    tuple(GL.mat_mundo.dot(p_c).flatten()),
                )
            )
        return tuple(triangles)

    @staticmethod
    def create_view_triangles(
        view_points: Tuple[Tuple[float]],
    ) -> Tuple[Tuple[Tuple[float]]]:
        """
        Create a tuple where each element is another tuple with 3 tuples.

        (
            (
                (a_x, a_y, a_z),
                (b_x, b_y, b_z),
                (c_x, c_y, c_z)
            ),
            (
                (a_x, a_y, a_z),
                (b_x, b_y, b_z),
                (c_x, c_y, c_z)
            ),
            ...
        )
        """

        triangles_screen = []
        for i in range(0, len(view_points), 3):
            triangles_screen.append(
                (view_points[i], view_points[i + 1], view_points[i + 2])
            )

        return tuple(triangles_screen)

    @staticmethod
    def is_inside_triangle(
        triangle: Tuple[Tuple[float]], point: Tuple[Tuple[float]]
    ) -> bool:
        """
        Determines if a given point is inside the given triangle.
        """
        p0, p1, p2 = triangle
        b0 = (point[0] - p0[0]) * (p1[1] - p0[1]) - (point[1] - p0[1]) * (p1[0] - p0[0])
        b1 = (point[0] - p1[0]) * (p2[1] - p1[1]) - (point[1] - p1[1]) * (p2[0] - p1[0])
        b2 = (point[0] - p2[0]) * (p0[1] - p2[1]) - (point[1] - p2[1]) * (p0[0] - p2[0])

        return (b0 >= 0) and (b1 >= 0) and (b2 >= 0)

    @staticmethod
    def color_triangle_pixels(
        triangles: Tuple[Tuple[Tuple[float]]], color: List[float]
    ):
        """
        Fills the framebuffer with the color of the triangles.
        """

        for t in triangles:
            point_a = t[0]
            point_b = t[1]
            point_c = t[2]

            # Calculate the box around the triangle.
            min_x = min(point_a[0], point_b[0], point_c[0])
            min_y = min(point_a[1], point_b[1], point_c[1])
            max_x = max(point_a[0], point_b[0], point_c[0])
            max_y = max(point_a[1], point_b[1], point_c[1])

            # For every pixel inside the box, check if it is inside the triangle.
            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    if GL.is_inside_triangle(t, (x, y)):
                        GL.framebuffer[x, y] = color

    @staticmethod
    def color_triangle_pixels_with_lighting(
        view_triangles: Tuple[Tuple[Tuple[float]]],
        world_triangles: Tuple[Tuple[Tuple[float]]],
        colors: Dict[str, List[float]],
    ):
        """
        Fills the framebuffer with the color of the triangles.
        """

        # print(f"[DrawTrianglesLight] {colors=}")
        # print(f"[DrawTrianglesLight] {GL.light_ambient_intensity=}")
        # print(f"[DrawTrianglesLight] {GL.light_color=}")
        # print(f"[DrawTrianglesLight] {GL.light_intensity=}")
        # print(f"[DrawTrianglesLight] {GL.light_dir=}")

        diffuse_color = (np.array(colors["diffuseColor"]) * 255).astype(int)
        emissive_color = (np.array(colors["emissiveColor"]) * 255).astype(int)
        specular_color = (np.array(colors["specularColor"]) * 255).astype(int)
        shininess: float = colors["shininess"]
        view_vector = np.array([0, 0, 1])

        ambient_color = GL.add_ambient_light(diffuse_color)

        for i in range(len(view_triangles)):
            wp_a = np.array(world_triangles[i][0][:3])
            wp_b = np.array(world_triangles[i][1][:3])
            wp_c = np.array(world_triangles[i][2][:3])
            # print(f"[DrawTrianglesLight] {wp_a=} {wp_b=} {wp_c=}")

            if GL.has_light:
                # Calculate the normal of the triangle, using the world points, and normalize it.
                normal: np.ndarray = np.cross(wp_a - wp_b, wp_a - wp_c)
                normal: np.ndarray = normal / np.linalg.norm(normal)
                # print(f"[DrawTrianglesLight] {normal=}")

                # Calculate the lambert color of the triangle.
                lambert: np.ndarray = GL.add_lambert(diffuse_color, normal)

            vp_a = view_triangles[i][0]
            vp_b = view_triangles[i][1]
            vp_c = view_triangles[i][2]

            # Calculate the box around the triangle.
            min_x = min(vp_a[0], vp_b[0], vp_c[0])
            min_y = min(vp_a[1], vp_b[1], vp_c[1])
            max_x = max(vp_a[0], vp_b[0], vp_c[0])
            max_y = max(vp_a[1], vp_b[1], vp_c[1])

            # For every pixel inside the box, check if it is inside the triangle.
            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    if GL.is_inside_triangle(view_triangles[i], (x, y)):
                        if GL.has_light:
                            half_vector = view_vector + GL.light_dir
                            half_vector = half_vector / np.linalg.norm(half_vector)
                            # print(f"[DrawTrianglesLight] {half_vector=}")
                            blinn_phong = GL.add_blinn_phong(
                                shininess, half_vector, specular_color, normal
                            )
                            color = np.clip(
                                (
                                    emissive_color
                                    + lambert
                                    + blinn_phong
                                    + ambient_color
                                ),
                                0,
                                255,
                            )
                            # print(f"[DrawTrianglesLight] {color=}")

                            GL.framebuffer[x, y] = color
                        else:
                            GL.framebuffer[x, y] = diffuse_color

    @staticmethod
    def triangleSet(point: List[float], colors: Dict[str, List[float]]):
        """
        Função usada para renderizar TriangleSet.

        Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        assim por diante.

        No TriangleSet os triângulos são informados individualmente, assim os três
        primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        triângulo, e assim por diante.

        O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet
        você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).
        """

        # print(f"[TriangleSet] {point=}")
        # print(f"[TriangleSet] {colors=}")

        world_triangles: Tuple[Tuple[Tuple[float]]] = GL.create_world_triangles(point)
        print(f"[TriangleSet] {world_triangles=}")

        view_triangles = GL.create_view_triangles(GL.prepare_points(point))
        print(f"[TriangleSet] {view_triangles=}")

        start = time.time()
        GL.color_triangle_pixels_with_lighting(view_triangles, world_triangles, colors)
        end = time.time()
        print(
            f"[TriangleSet] Time taken to calculate the pixels inside: {end - start}s"
        )

        start = time.time()
        GL.supersampling_2x()
        end = time.time()
        print(f"[TriangleSet] Time taken to Super Sample 2X: {end - start}s")

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        GL.fieldOfView = fieldOfView

        # Criando as matrizes de transformação da câmera de acordo com os dados coletados nesse frame.
        GL.trans_mat_cam = np.array(
            [
                [1, 0, 0, position[0]],
                [0, 1, 0, position[1]],
                [0, 0, 1, position[2]],
                [0, 0, 0, 1],
            ]
        )

        if orientation:
            if orientation[0] > 0:
                # Rotação em x
                GL.orient_mat_cam = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(orientation[3]), -np.sin(orientation[3]), 0],
                        [0, np.sin(orientation[3]), np.cos(orientation[3]), 0],
                        [0, 0, 0, 1],
                    ]
                )
            elif orientation[1] > 0:
                # Rotação em y
                GL.orient_mat_cam = np.array(
                    [
                        [np.cos(orientation[3]), 0, np.sin(orientation[3]), 0],
                        [0, 1, 0, 0],
                        [-np.sin(orientation[3]), 0, np.cos(orientation[3]), 0],
                        [0, 0, 0, 1],
                    ]
                )
            else:
                # Rotação em z
                GL.orient_mat_cam = np.array(
                    [
                        [np.cos(orientation[3]), -np.sin(orientation[3]), 0, 0],
                        [np.sin(orientation[3]), np.cos(orientation[3]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo em alguma estrutura de pilha.

        # Criando as matrizes de transformação de acordo com os dados coletados nesse frame.
        if translation:
            GL.trans_mat = np.array(
                [
                    [1, 0, 0, translation[0]],
                    [0, 1, 0, translation[1]],
                    [0, 0, 1, translation[2]],
                    [0, 0, 0, 1],
                ]
            )

        if scale:
            GL.scale_mat = np.array(
                [
                    [scale[0], 0, 0, 0],
                    [0, scale[1], 0, 0],
                    [0, 0, scale[2], 0],
                    [0, 0, 0, 1],
                ]
            )

        if rotation:
            if rotation[0] > 0:
                # Rotação em x
                GL.rot_mat = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(rotation[3]), -np.sin(rotation[3]), 0],
                        [0, np.sin(rotation[3]), np.cos(rotation[3]), 0],
                        [0, 0, 0, 1],
                    ]
                )
            elif rotation[1] > 0:
                # Rotação em y
                GL.rot_mat = np.array(
                    [
                        [np.cos(rotation[3]), 0, np.sin(rotation[3]), 0],
                        [0, 1, 0, 0],
                        [-np.sin(rotation[3]), 0, np.cos(rotation[3]), 0],
                        [0, 0, 0, 1],
                    ]
                )
            else:
                # Rotação em z
                GL.rot_mat = np.array(
                    [
                        [np.cos(rotation[3]), -np.sin(rotation[3]), 0, 0],
                        [np.sin(rotation[3]), np.cos(rotation[3]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]
                )

        ans = GL.trans_mat.dot(GL.rot_mat).dot(GL.scale_mat)

        if bool(GL.world_transform_stack) == True:
            ans = np.dot(
                GL.world_transform_stack[len(GL.world_transform_stack) - 1], ans
            )

        GL.world_transform_stack += [ans]
        GL.mat_mundo: np.array = ans

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Saindo de Transform")
        if len(GL.world_transform_stack) > 0:
            GL.world_transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        print(f"[TriangleStripSet] {point=}")
        print(f"[TriangleStripSet] {stripCount=}")
        print(f"[TriangleStripSet] {colors=}")

        # Criando uma lista de pontos, onde cada ponto é uma tupla.
        points: List[Tuple[float]] = []
        for i in range(len(point) // 3):
            points += [(point[i * 3], point[i * 3 + 1], point[i * 3 + 2])]

        # Criando a lista de triângulos, seguindo a ordem de stripCount,
        # onde cada triângulo é uma tupla de pontos.
        world_triangles: List[Tuple[Tuple[float]]] = []
        for i in range(stripCount[0] - 2):
            world_triangles.append((points[i], points[i + 1], points[i + 2]))
        print(f"[TriangleStripSet] {world_triangles=}")

        prepared_points = GL.prepare_points(point)
        view_triangles: List[Tuple[Tuple[float]]] = []
        for i in range(stripCount[0] - 2):
            view_triangles.append(
                (prepared_points[i], prepared_points[i + 1], prepared_points[i + 2])
            )
        print(f"[TriangleStripSet] {view_triangles=}")

        start = time.time()
        GL.color_triangle_pixels(view_triangles, np.array(colors["diffuseColor"]) * 255)
        end = time.time()
        print(
            f"[TriangleSet] Time taken to calculate the pixels inside: {end - start}s"
        )

        start = time.time()
        GL.supersampling_2x()
        end = time.time()
        print(f"[TriangleSet] Time taken to Super Sample 2X: {end - start}s")

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        # print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        print(f"[IndexedTriangleStripSet] {point=}")
        print(f"[IndexedTriangleStripSet] {index=}")
        print(f"[IndexedTriangleStripSet] {colors=}")

        # Criando uma lista de pontos, onde cada ponto é uma tupla.
        points: List[Tuple[float]] = []
        for i in range(len(point) // 3):
            points += [(point[i * 3], point[i * 3 + 1], point[i * 3 + 2])]

        return
        triangle_points = GL.prepare_points(point * 2)

        color = None
        if GL.has_light:
            ambient = GL.add_ambient_light(colors["diffuseColor"])
            object_base_color = np.array(colors["emissiveColor"])

        else:
            color = np.array(colors["diffuseColor"]).astype(int) * 255

        for i in range(len(index) - 3):
            if i % 2 == 0:
                x0, y0 = int(triangle_points[index[i]][0]), int(
                    triangle_points[index[i]][1]
                )
                x1, y1 = int(triangle_points[index[i + 1]][0]), int(
                    triangle_points[index[i + 1]][1]
                )
                x2, y2 = int(triangle_points[index[i + 2]][0]), int(
                    triangle_points[index[i + 2]][1]
                )
            else:
                x2, y2 = int(triangle_points[index[i]][0]), int(
                    triangle_points[index[i]][1]
                )
                x1, y1 = int(triangle_points[index[i + 1]][0]), int(
                    triangle_points[index[i + 1]][1]
                )
                x0, y0 = int(triangle_points[index[i + 2]][0]), int(
                    triangle_points[index[i + 2]][1]
                )

            if GL.has_light:
                lambert = GL.add_lambert(colors["diffuseColor"], [0, -0.577, -0.577])

            # Calcula se o ponto (x, y) está acima, abaixo, ou na linha descrita por P0 -> P1.
            L0 = lambda x, y: (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)

            # Calcula se o ponto (x, y) está acima, abaixo, ou na linha descrita por P1 -> P2.
            L1 = lambda x, y: (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

            # Calcula se o ponto (x, y) está acima, abaixo, ou na linha descrita por P2 -> P0.
            L2 = lambda x, y: (x - x2) * (y0 - y2) - (y - y2) * (x0 - x2)

            # Determina se o ponto está dentro do triângulo ou não.
            inside = lambda x, y: L0(x, y) >= 0 and L1(x, y) >= 0 and L2(x, y) >= 0

            for si in range(GL.width * 2):
                for sj in range(GL.height * 2):
                    if inside(si + 0.5, sj + 0.5):
                        if GL.has_light:
                            half_vector = GL.light_dir + GL.view_vector
                            half_vector = half_vector / np.linalg.norm(N)
                            blinn_phong = GL.add_blinn_phong(
                                colors["shininess"],
                                half_vector,
                                colors["specularColor"],
                            )
                            color = (
                                object_base_color + ambient + lambert + phong
                            ) * 255

                        GL.framebuffer[si, sj] = color

        # GL.supersampling_2x()

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        x = size[0]
        y = size[1]
        z = size[2]

        bottom_front_left = (-x, -y, z)
        bottom_front_right = (x, -y, z)
        bottom_back_left = (x, y, z)
        bottom_back_right = (-x, y, z)
        top_front_left = (-x, y, -z)
        top_front_right = (x, y, -z)
        top_back_left = (-x, -y, -z)
        top_back_right = (x, -y, -z)

        point = np.array(
            [
                bottom_front_left,
                bottom_front_right,
                bottom_back_left,
                bottom_back_left,
                bottom_back_right,
                bottom_front_left,
                top_back_left,
                bottom_front_left,
                bottom_back_right,
                bottom_back_right,
                top_front_left,
                top_back_left,
                top_back_right,
                top_back_left,
                top_front_left,
                top_front_left,
                top_front_right,
                top_back_right,
                top_back_right,
                top_front_right,
                bottom_front_right,
                bottom_front_right,
                top_front_right,
                bottom_back_left,
                top_front_right,
                top_front_left,
                bottom_back_right,
                bottom_back_right,
                bottom_back_left,
                top_front_right,
                top_back_right,
                top_back_left,
                bottom_front_left,
                bottom_front_left,
                bottom_front_right,
                top_back_right,
            ]
        ).flatten()

        GL.triangleSet(point, colors)

    @staticmethod
    def indexedFaceSet(
        coord,
        coordIndex,
        colorPerVertex,
        color,
        colorIndex,
        texCoord,
        texCoordIndex,
        colors,
        current_texture,
    ):
        """Função usada para renderizar IndexedFaceSet."""
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        triangle_points = GL.prepare_points(coord * 2)

        if texCoord and texCoordIndex and current_texture:
            texture = gpu.GPU.load_texture(current_texture[0])

        for i in range(len(coordIndex) - 3):
            if colorPerVertex and color and colorIndex:
                offset_0 = (colorIndex[i + 0]) * 3
                offset_1 = (colorIndex[i + 1]) * 3
                offset_2 = (colorIndex[i + 2]) * 3

                vertex_color_0 = (
                    int(color[offset_0 + 0] * 255),
                    int(color[offset_0 + 1] * 255),
                    int(color[offset_0 + 2] * 255),
                )

                vertex_color_1 = (
                    int(color[offset_1 + 0] * 255),
                    int(color[offset_1 + 1] * 255),
                    int(color[offset_1 + 2] * 255),
                )

                vertex_color_2 = (
                    int(color[offset_2 + 0] * 255),
                    int(color[offset_2 + 1] * 255),
                    int(color[offset_2 + 2] * 255),
                )

            if i % 2 == 0:
                x0, y0, z0 = (
                    (triangle_points[coordIndex[i + 0]][0]),
                    (triangle_points[coordIndex[i + 0]][1]),
                    triangle_points[coordIndex[i + 0]][2][0],
                )
                x1, y1, z1 = (
                    (triangle_points[coordIndex[i + 1]][0]),
                    (triangle_points[coordIndex[i + 1]][1]),
                    triangle_points[coordIndex[i + 1]][2][0],
                )
                x2, y2, z2 = (
                    (triangle_points[coordIndex[i + 2]][0]),
                    (triangle_points[coordIndex[i + 2]][1]),
                    triangle_points[coordIndex[i + 2]][2][0],
                )

            else:
                x2, y2, z2 = (
                    (triangle_points[coordIndex[i + 0]][0]),
                    (triangle_points[coordIndex[i + 0]][1]),
                    triangle_points[coordIndex[i + 0]][2][0],
                )
                x1, y1, z1 = (
                    (triangle_points[coordIndex[i + 1]][0]),
                    (triangle_points[coordIndex[i + 1]][1]),
                    triangle_points[coordIndex[i + 1]][2][0],
                )
                x0, y0, z0 = (
                    (triangle_points[coordIndex[i + 2]][0]),
                    (triangle_points[coordIndex[i + 2]][1]),
                    triangle_points[coordIndex[i + 2]][2][0],
                )

            z0 = 1 / z0
            z1 = 1 / z1
            z2 = 1 / z2

            # Calcula se o ponto (x, y) está acima, abaixo, ou na linha descrita por P0 -> P1.
            L0 = lambda x, y: (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)

            # Calcula se o ponto (x, y) está acima, abaixo, ou na linha descrita por P1 -> P2.
            L1 = lambda x, y: (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

            # Calcula se o ponto (x, y) está acima, abaixo, ou na linha descrita por P2 -> P0.
            L2 = lambda x, y: (x - x2) * (y0 - y2) - (y - y2) * (x0 - x2)

            # Determina se o ponto está dentro do triângulo ou não.
            inside = lambda x, y: L0(x, y) >= 0 and L1(x, y) >= 0 and L2(x, y) >= 0

            alpha_denominator = -(x0 - x1) * (y2 - y1) + (y0 - y1) * (x2 - x1)
            betha_denominator = -(x1 - x2) * (y0 - y2) + (y1 - y2) * (x0 - x2)

            for si in range(GL.width * 2):
                for sj in range(GL.height * 2):
                    if inside(si + 0.5, sj + 0.5):
                        alpha = (
                            -(si - x1) * (y2 - y1) + (sj - y1) * (x2 - x1)
                        ) / alpha_denominator
                        betha = (
                            -(si - x2) * (y0 - y2) + (sj - y2) * (x0 - x2)
                        ) / betha_denominator
                        gamma = 1 - alpha - betha

                        z = z0 * alpha + z2 * gamma + z1 * betha

                        # print(f"{alpha=}, {betha=}, {gamma=}, {z0=}, {z1=}, {z2=}, {z=}")

                        if colorPerVertex and color and colorIndex:
                            c = (
                                (
                                    vertex_color_0[0] * alpha
                                    + vertex_color_1[0] * betha
                                    + vertex_color_2[0] * gamma
                                ),
                                (
                                    vertex_color_0[1] * alpha
                                    + vertex_color_1[1] * betha
                                    + vertex_color_2[1] * gamma
                                ),
                                (
                                    vertex_color_0[2] * alpha
                                    + vertex_color_1[2] * betha
                                    + vertex_color_2[2] * gamma
                                ),
                            )

                        elif texCoord and texCoordIndex and current_texture:
                            offset_1 = texCoordIndex[i + 0] * 2
                            offset_2 = texCoordIndex[i + 1] * 2
                            offset_3 = texCoordIndex[i + 2] * 2

                            uv_vertex_1 = [texCoord[offset_1], texCoord[offset_1 + 1]]
                            uv_vertex_2 = [texCoord[offset_2], texCoord[offset_2 + 1]]
                            uv_vertex_3 = [texCoord[offset_3], texCoord[offset_3 + 1]]

                            uv_1 = [t * z0 for t in uv_vertex_1]
                            uv_2 = [t * z1 for t in uv_vertex_2]
                            uv_3 = [t * z2 for t in uv_vertex_3]

                            u = (
                                (uv_1[0] * alpha + uv_2[0] * betha + uv_3[0] * gamma)
                                / z
                            ) * (texture.shape[0] - 1)
                            v = -(
                                (
                                    (
                                        uv_1[1] * alpha
                                        + uv_2[1] * betha
                                        + uv_3[1] * gamma
                                    )
                                    / z
                                )
                                * (texture.shape[1] - 1)
                            )

                            # print(f"{u=}, {v=}")

                            img_color = texture[int(v)][int(u)]
                            c = tuple(img_color[:3])

                        else:
                            c = (255, 255, 255)

                        GL.framebuffer[si, sj] = c

        GL.supersampling_2x()

    @staticmethod
    def view_point(fovx, near, far, width, height):
        fovy = 2 * np.arctan(
            np.tan(fovx / 2) * height / (height ** 2 + width ** 2) ** 0.5
        )
        top = near * np.tan(fovy)
        right = top * width / height

        return np.array(
            [
                [near / right, 0, 0, 0],
                [0, near / top, 0, 0],
                [
                    0,
                    0,
                    -((far + near) / (far - near)),
                    ((-2 * far * near) / (far - near)),
                ],
                [0, 0, -1, 0],
            ]
        )

    @staticmethod
    def prepare_points(points) -> Tuple[Tuple[float]]:
        GL.lookAt = np.linalg.inv(GL.orient_mat_cam).dot(
            np.linalg.inv(GL.trans_mat_cam)
        )

        screen_points = []
        for i in range(0, len(points), 3):
            current_point = np.array(
                [[points[i]], [points[i + 1]], [points[i + 2]], [1]]
            )

            # Transformação do ponto para coordenadas de tela
            current_point = GL.mat_mundo.dot(current_point)
            current_point = GL.lookAt.dot(current_point)

            # Leva o ponto para as coordenadas de perspectiva
            current_point = GL.view_point(
                GL.fieldOfView, GL.near, GL.far, GL.width * 2, GL.height * 2
            ).dot(current_point)
            actual_z = current_point[2][0]
            current_point /= current_point[3][0]
            current_point = GL.screen_mat.dot(current_point)
            current_point[2][0] = actual_z
            screen_points.append(tuple(current_point.flatten()))

        return tuple(screen_points)

    @staticmethod
    def supersampling_2x():
        """
        Supersamples the framebuffer by 2x.
        """

        for i in range(0, GL.width * 2, 2):
            for j in range(0, GL.height * 2, 2):
                pixel1 = GL.framebuffer[i, j]
                pixel2 = GL.framebuffer[i + 1, j]
                pixel3 = GL.framebuffer[i, j + 1]
                pixel4 = GL.framebuffer[i + 1, j + 1]

                new_color = (pixel1 + pixel2 + pixel3 + pixel4) // 4

                if new_color[0] > 0 or new_color[1] > 0 or new_color[2] > 0:
                    gpu.GPU.set_pixel(
                        int(i / 2), int(j / 2), new_color[0], new_color[1], new_color[2]
                    )

    @staticmethod
    def sphere(radius: float, colors: Dict[str, List[float]]):
        """
        Função usada para renderizar Esferas.

        A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        raio da esfera que está sendo criada. Para desenha essa esfera você vai
        precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        os triângulos.
        """

        local_triangles: Tuple[Tuple[Tuple[float]]] = GL.sphere_triangles(
            radius, step=15
        )
        # print(f"[Sphere] {local_triangles=}")

        local_points = np.array(local_triangles).flatten()
        world_triangles = GL.create_world_triangles(local_points)
        # print(f"[Sphere] {world_triangles=}")

        view_triangles: Tuple[Tuple[Tuple[float]]] = GL.create_view_triangles(
            GL.prepare_points(local_points)
        )
        # print(f"[Sphere] {view_triangles=}")

        start = time.time()
        GL.color_triangle_pixels_with_lighting(view_triangles, world_triangles, colors)
        end = time.time()
        print(
            f"[TriangleSet] Time taken to calculate the pixels inside: {end - start}s"
        )

        start = time.time()
        GL.supersampling_2x()
        end = time.time()
        print(f"[TriangleSet] Time taken to Super Sample 2X: {end - start}s")

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal
        GL.has_light = headlight
        if headlight:
            GL.light_ambient_intensity = 0
            GL.light_color = np.array((1, 1, 1))
            GL.light_intensity = 1
            GL.light_dir = -1 * np.array([0, 0, -1])

            print(f"[NavigationInfo] {GL.light_ambient_intensity=}")
            print(f"[NavigationInfo] {GL.light_color=}")
            print(f"[NavigationInfo] {GL.light_intensity=}")
            print(f"[NavigationInfo] {GL.light_dir=}")

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        # print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        # print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        # print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal
        # print("AAAAAAAAAA")

        GL.has_light = True
        GL.light_ambient_intensity = ambientIntensity
        GL.light_color = np.array(color)
        GL.light_intensity = intensity
        GL.light_dir = -1 * np.array(direction)

        print(f"[DirectionalLight] {GL.light_ambient_intensity=}")
        print(f"[DirectionalLight] {GL.light_color=}")
        print(f"[DirectionalLight] {GL.light_intensity=}")
        print(f"[DirectionalLight] {GL.light_dir=}")

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        # print("PointLight : color = {0}".format(color)) # imprime no terminal
        # print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        # print("PointLight : location = {0}".format(location)) # imprime no terminal
        # print("BBBBBBBBBBBBBBBB")

        GL.has_light = True
        GL.light_ambient_intensity = ambientIntensity
        GL.light_color = np.array(color) / np.linalg.norm(color)
        GL.light_intensity = intensity
        GL.location = location

        print(f"[PointLight] {GL.light_ambient_intensity=}")
        print(f"[PointLight] {GL.light_color=}")
        print(f"[PointLight] {GL.light_intensity=}")
        print(f"[PointLight] {GL.light_dir=}")

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color))  # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "TimeSensor : cycleInterval = {0}".format(cycleInterval)
        )  # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = (
            time.time()
        )  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print(
            "SplinePositionInterpolator : key = {0}".format(key)
        )  # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]

        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key))  # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    @staticmethod
    def add_blinn_phong(shine, half_vector, specular_color, normal):
        half_dot_normal = max(0, np.dot(half_vector, normal))
        specular_exponent = int(shine * 128)
        specular_light = (
            (half_dot_normal ** specular_exponent) * GL.light_intensity * specular_color
        )
        # print(f"[BlinnPhong] {specular_light=}")
        return specular_light

    @staticmethod
    def add_lambert(obj_difuse_color, normal):
        """
        Calculates the lambert color of the triangle.
        """
        diff = np.dot(normal, GL.light_dir)
        lambert = diff * GL.light_intensity * obj_difuse_color
        # print(f"[Lighting] {lambert=}")
        return lambert

    @staticmethod
    def add_ambient_light(obj_difuse_color):
        return np.array(obj_difuse_color) * GL.light_ambient_intensity

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""

    @staticmethod
    def sphere_triangles(radius, step=10):
        """
        Create a list of triangles on a sphere
        """
        import math

        triangles = []
        for theta in range(0, 360, step):
            for phi in range(0, 360, step):
                r_theta = math.radians(theta)
                cos_r_theta = math.cos(r_theta)
                sin_r_theta = math.sin(r_theta)
                r_theta_step = math.radians(theta + step)
                cos_r_theta_step = math.cos(r_theta_step)
                sin_r_theta_step = math.sin(r_theta_step)

                r_phi = math.radians(phi)
                cos_r_phi = math.cos(r_phi)
                sin_r_phi = math.sin(r_phi)
                r_phi_step = math.radians(phi + step)
                cos_r_phi_step = math.cos(r_phi_step)
                sin_r_phi_step = math.sin(r_phi_step)

                p1 = (
                    radius * cos_r_theta * cos_r_phi,
                    radius * sin_r_theta * cos_r_phi,
                    radius * sin_r_phi,
                )
                p2 = (
                    radius * cos_r_theta * cos_r_phi_step,
                    radius * sin_r_theta * cos_r_phi_step,
                    radius * sin_r_phi_step,
                )
                p3 = (
                    radius * cos_r_theta_step * cos_r_phi,
                    radius * sin_r_theta_step * cos_r_phi,
                    radius * sin_r_phi,
                )
                p4 = (
                    radius * cos_r_theta_step * cos_r_phi_step,
                    radius * sin_r_theta_step * cos_r_phi_step,
                    radius * sin_r_phi_step,
                )
                triangles.append((p1, p2, p3))
                triangles.append((p2, p4, p3))
        return tuple(triangles)
